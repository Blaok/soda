#include "schedules.h"

#include <csignal>
#include <cstdlib>

#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include <glog/logging.h>

using std::make_shared;
using std::numeric_limits;
using std::shared_ptr;
using std::string;
using std::vector;

template<typename RAttr, typename AAttr>
Schedules<RAttr, AAttr>::Schedules(
    const Context& context,
    const vector<RAttr>& rattr, const vector<AAttr>* aattr,
    AAttr num_ops, AAttr offset, AAttr max_cost)
    : context_(context), rattr_(&rattr), aattr_(aattr),
      num_ops_(num_ops == AAttr(-1) ? rattr.size() - 1 : num_ops),
      offset_(offset), max_cost_(max_cost == AAttr(-1) ? num_ops_ : max_cost) {}

template<typename RAttr, typename AAttr>
typename Schedules<RAttr, AAttr>::Operation
Schedules<RAttr, AAttr>::MakeOperation(const Slice& slice,
                                       const BRepr& brepr) const {
  RAttr offset = (*rattr_)[slice.first];
  Operation operation;
  operation.rattrs.resize(slice.second - slice.first);
  const RAttr* src_rattr = &rattr_->at(slice.first);
  for (auto & dst_rattr : operation.rattrs) {
    dst_rattr = *(src_rattr++) - offset;
  }
  operation.aattr = aattr_ == nullptr ? nullptr : &aattr_->at(slice.first);
  operation.brepr = &brepr;
  return operation;
}

template<typename RAttr, typename AAttr>
const typename Schedules<RAttr, AAttr>::ScheduleVec&
Schedules<RAttr, AAttr>::GetSchedules(AAttr num_ops, AAttr offset) {
  if (auto schedules = GetCache(num_ops, offset)) {
    return schedules->schedules_;
  }
  const auto key = std::make_pair(num_ops, offset);
  auto acquire_task = [&key, this]() -> bool {
    WriteLock lock(*context_.tasks_mtx);
    bool acquired = false;
    if (context_.tasks->count(key) == 0) {
      acquired = true;
      (*context_.tasks)[key] = false;
    }
    return acquired;
  };
  if (acquire_task()) {
    auto schedules = make_shared<Schedules>(
        context_, *rattr_, aattr_, num_ops, offset, max_cost_ + 1);
    schedules->Generate();
    SetCache(num_ops, offset, schedules);
    (*context_.tasks)[key].store(true);
    return schedules->schedules_;
  }
  while (!(*context_.tasks)[key].load()) {
    std::this_thread::yield();
  }
  return GetCache(num_ops, offset)->schedules_;
}

template<typename RAttr, typename AAttr>
const typename Schedules<RAttr, AAttr>::ScheduleVec&
Schedules<RAttr, AAttr>::Generate() {
  const AAttr n = num_ops_;
  const AAttr k = offset_;
  if (schedules_.size() > 0) {
    return schedules_;
  }
  if (n == 0) {
    schedules_.emplace_back(make_shared<const Schedule>(
        make_shared<const BRepr>(BRepr({1})), make_shared<const Operations>(),
        make_shared<const OperationSet>(), rattr_, aattr_));
    return schedules_;
  }
  const auto vec = context_.Shuffle(n);
#pragma omp parallel for schedule(dynamic) firstprivate(n, k)
  for (AAttr i = 0; i < n; ++i) {
    AAttr m = vec[i];
    context_.VisitLoop(1);
    for (const auto& prefix : GetSchedules(m, k)) {
      context_.VisitLoop(2);
      Operations prefix_operations;
      OperationSet prefix_operation_set;
      // Only slices with len > 1 are operations.
      if (m > 0) {
        prefix_operations.emplace_back(k, k + m + 1);
        prefix_operation_set.insert(MakeOperation(*prefix_operations.rbegin(),
                                                  *prefix->brepr));
        if (AAttr(prefix_operation_set.size()) >= max_cost_) {
          continue;
        }
      }
      prefix_operations.insert(prefix_operations.end(),
                               prefix->operations->begin(),
                               prefix->operations->end());
      prefix_operation_set.insert(prefix->operation_set->begin(),
                                  prefix->operation_set->end());
      if (AAttr(prefix_operation_set.size()) >= max_cost_) {
        continue;
      }
      for (const auto& suffix : GetSchedules(n - m - 1, k + m + 1)) {
        context_.VisitLoop(3);
        auto operations = new Operations(prefix_operations);
        auto operation_set = new OperationSet(prefix_operation_set);
        if (n > m + 1) {
          operations->emplace_back(k + m + 1, k + n + 1);
          operation_set->insert(MakeOperation(*operations->rbegin(),
                                              *suffix->brepr));
          if (AAttr(operation_set->size()) >= max_cost_) {
            delete operations;
            delete operation_set;
            continue;
          }
        }
        operations->insert(operations->end(), suffix->operations->begin(),
                           suffix->operations->end());
        operation_set->insert(suffix->operation_set->begin(),
                              suffix->operation_set->end());
        if (AAttr(operation_set->size()) >= max_cost_) {
          delete operations;
          delete operation_set;
          continue;
        }
        auto brepr = new BRepr;
        brepr->reserve(prefix->brepr->size() + suffix->brepr->size() + 1);
        brepr->push_back(0);
        brepr->insert(brepr->end(), prefix->brepr->begin(),
                      prefix->brepr->end());
        brepr->insert(brepr->end(), suffix->brepr->begin(),
                      suffix->brepr->end());
#pragma omp critical
        {
          if (max_cost_ > AAttr(operation_set->size())) {
            max_cost_ = operation_set->size();
          }
          schedules_.emplace_back(make_shared<const Schedule>(
              shared_ptr<const BRepr>(brepr),
              shared_ptr<const Operations>(operations),
              shared_ptr<const OperationSet>(operation_set), rattr_, aattr_));
        }
      }
    }
  }
  return schedules_;
}

template<typename RAttr, typename AAttr>
typename Schedules<RAttr, AAttr>::Schedule Schedules<RAttr, AAttr>::Best() {
  Schedule best;
  for (const auto& schedule : Generate()) {
    if (schedule->cost < best.cost) { best = *schedule; }
  }
  return best;
}

void ContextBase::PrintStats(std::ostream& stream) const {
  ReadLock lock(*stat_mtx_);
  stream << "loops: | L1: " << LoopTripCount(1) << " | L2: "
         << LoopTripCount(2) << " | L3: " << LoopTripCount(3) << " |\n"
         << "cache: | hit: " << CacheHit() << " | miss: " << CacheMiss()
         << " | hit rate: "
         << float(CacheHit()) / (CacheHit() + CacheMiss()) * 100 << " % |\n";
}

string ToString(const BRepr& brepr) {
  string result;
  result.reserve(brepr.size());
  for (auto bit : brepr) {
    result.push_back(bit ? '1' : '0');
  }
  return result;
}

template<typename RAttr, typename AAttr>
void TemporalCseKernel(const int64_t* rattr_ptr, const int64_t* aattr_ptr,
                 uint64_t n, uint64_t* cost, char* brepr,
                 uint64_t* operations, uint64_t* stat, uint64_t* config) {
  vector<RAttr> rattr(n);
  vector<AAttr>* aattr = nullptr;
  for (size_t i = 0; i < n; ++i) {
    rattr[i] = rattr_ptr[i];
  }
  if (aattr_ptr != nullptr) {
    aattr = new vector<AAttr>(n);
    for (size_t i = 0; i < n; ++i) {
      aattr->at(i) = aattr_ptr[i];
    }
  }
  LOG(INFO) << "invoke native temporal CSE with " << sizeof(RAttr) * 8
            << "-bit RAttr and " << sizeof(AAttr) * 8 << "-bit AAttr";
  if (config != nullptr) {
    Schedules<RAttr, AAttr>::Context::exploration_order =
      static_cast<tcse::ExplorationOrder>(config[0]);
    if (config[0] == tcse::kRandom) {
      Schedules<RAttr, AAttr>::Context::seed = config[1];
    }
  }
  auto cache = make_shared<typename Schedules<RAttr, AAttr>::CacheType>();
  auto schedules = Schedules<RAttr, AAttr>(cache, rattr, aattr);
  auto best = schedules.Best();
  *cost = best.cost;
  for (size_t i = 0; i < best.brepr->size(); ++i) {
    brepr[i] = best.brepr->at(i) ? '1' : '0';
  }
  brepr[best.brepr->size()] = '\0';
  for (size_t i = 0; i < best.operations->size(); ++i) {
    operations[i * 2] = best.operations->at(i).first;
    operations[i * 2 + 1] = best.operations->at(i).second;
  }
  schedules.Export(stat);
  LOG(INFO) << "finish native temporal CSE";
}

extern "C" {

void TemporalCse(const int64_t* rattr_ptr, const int64_t* aattr_ptr,
                 uint64_t n, uint64_t* cost, char* brepr, uint64_t* operations,
                 uint64_t* stat, uint64_t* config) {
  auto handler = [](int signum) { exit(signum); };
  signal(SIGINT, handler);
  google::InitGoogleLogging("libtemporal-cse");
  FLAGS_logtostderr = true;
  const int64_t max_rattr = rattr_ptr[n - 1];
  LOG(INFO) << "max relative attribute: " << max_rattr;
  LOG(INFO) << "max absolute attribute: " << n;
  if (max_rattr < numeric_limits<uint8_t>::max()) {
    if (n < numeric_limits<uint8_t>::max()) {
      TemporalCseKernel<uint8_t, uint8_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else if (n < numeric_limits<uint16_t>::max()) {
      TemporalCseKernel<uint8_t, uint16_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else if (n < numeric_limits<uint32_t>::max()) {
      TemporalCseKernel<uint8_t, uint32_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else {
      TemporalCseKernel<uint8_t, uint64_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    }
  } else if (max_rattr < numeric_limits<uint16_t>::max()) {
    if (n < numeric_limits<uint8_t>::max()) {
      TemporalCseKernel<uint16_t, uint8_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else if (n < numeric_limits<uint16_t>::max()) {
      TemporalCseKernel<uint16_t, uint16_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else if (n < numeric_limits<uint32_t>::max()) {
      TemporalCseKernel<uint16_t, uint32_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else {
      TemporalCseKernel<uint16_t, uint64_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    }
  } else if (max_rattr < numeric_limits<uint32_t>::max()) {
    if (n < numeric_limits<uint8_t>::max()) {
      TemporalCseKernel<uint32_t, uint8_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else if (n < numeric_limits<uint16_t>::max()) {
      TemporalCseKernel<uint32_t, uint16_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else if (n < numeric_limits<uint32_t>::max()) {
      TemporalCseKernel<uint32_t, uint32_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else {
      TemporalCseKernel<uint32_t, uint64_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    }
  } else {
    if (n < numeric_limits<uint8_t>::max()) {
      TemporalCseKernel<uint64_t, uint8_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else if (n < numeric_limits<uint16_t>::max()) {
      TemporalCseKernel<uint64_t, uint16_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else if (n < numeric_limits<uint32_t>::max()) {
      TemporalCseKernel<uint64_t, uint32_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    } else {
      TemporalCseKernel<uint64_t, uint64_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat, config);
    }
  }
}

}   // extern "C"
