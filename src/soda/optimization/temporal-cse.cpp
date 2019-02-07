#include "schedules.h"

#include <iostream>
#include <limits>
#include <memory>
#include <vector>

using std::make_shared;
using std::nullptr_t;
using std::numeric_limits;
using std::shared_ptr;
using std::string;
using std::vector;

inline uint64_t RangeFromMiddle(uint64_t n, uint64_t i) {
  if (n % 2 == 0) {
    if (i % 2 == 0) { return n / 2 - i / 2 - 1; }
    return n / 2 + i / 2;
  }
  if (i == 0) { return n / 2; }
  if (i % 2 == 1) { return n / 2 - (i + 1) / 2; }
  return n / 2 + (i + 1) / 2;
}

template<typename RAttr, typename AAttr>
Schedules<RAttr, AAttr>::Schedules(
    const vector<RAttr>& rattr, const vector<AAttr>* aattr,
    shared_ptr<CacheType> cache, AAttr num_ops, AAttr offset,
    shared_ptr<StatType> stat, AAttr max_cost)
    : ScheduleBase(
        stat == nullptr ? shared_ptr<StatType>(new StatType({})) : stat),
      rattr_(&rattr), aattr_(aattr), cache_(cache),
      num_ops_(num_ops == -1 ? rattr.size() - 1 : num_ops), offset_(offset),
      max_cost_(max_cost == -1 ? num_ops_ : max_cost) {}

template<typename RAttr, typename AAttr>
typename Schedules<RAttr, AAttr>::Operation
Schedules<RAttr, AAttr>::MakeOperation(const Slice& slice, const BRepr& brepr) {
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
  if (Cache() != nullptr) {
    if (Cache()->count(num_ops)) {
      if ((*Cache())[num_ops].count(offset)) {
        HitCache();
        return (*Cache())[num_ops][offset]->schedules_;
      }
    }
  }
  MissCache();
  auto schedules = make_shared<Schedules>(
      *rattr_, aattr_, Cache(), num_ops, offset, stat_, max_cost_ + 1);
  if (Cache() != nullptr) {
    (*Cache())[num_ops][offset] = schedules;
  }
  return schedules->Generate();
}

template<typename RAttr, typename AAttr>
const typename Schedules<RAttr, AAttr>::ScheduleVec&
Schedules<RAttr, AAttr>::Generate() {
  AAttr n = num_ops_;
  AAttr k = offset_;
  if (schedules_.size() > 0) {
    return schedules_;
  }
  if (n == 0) {
    schedules_.emplace_back(make_shared<const Schedule>(
        make_shared<const BRepr>(BRepr({1})), make_shared<const Operations>(),
        make_shared<const OperationSet>(), rattr_, aattr_));
    return schedules_;
  }
  for (AAttr i = 0; i < n; ++i) {
    AAttr m = RangeFromMiddle(n, i);
    VisitLoop(1);
    for (const auto& prefix : GetSchedules(m, k)) {
      VisitLoop(2);
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
        VisitLoop(3);
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
        max_cost_ = operation_set->size();
        auto brepr = new BRepr;
        brepr->reserve(prefix->brepr->size() + suffix->brepr->size() + 1);
        brepr->push_back(0);
        brepr->insert(brepr->end(), prefix->brepr->begin(),
                      prefix->brepr->end());
        brepr->insert(brepr->end(), suffix->brepr->begin(),
                      suffix->brepr->end());
        schedules_.emplace_back(make_shared<const Schedule>(
            shared_ptr<const BRepr>(brepr),
            shared_ptr<const Operations>(operations),
            shared_ptr<const OperationSet>(operation_set), rattr_, aattr_));
      }
    }
  }
  return schedules_;
}

template<typename RAttr, typename AAttr>
typename Schedules<RAttr, AAttr>::Schedule Schedules<RAttr, AAttr>::Best() {
  Schedule best;
  for (const auto& schedule : Generate()) {
    //if (schedule->cost < best.cost) { best = *schedule; }
    best = *schedule;
  }
  return best;
}

void ScheduleBase::PrintStats(std::ostream& stream) {
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
                 uint64_t* operations, uint64_t* stat) {
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
  auto cache = make_shared<typename Schedules<RAttr, AAttr>::CacheType>();
  auto schedules = Schedules<RAttr, AAttr>(rattr, aattr, cache);
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
  stat[0] = schedules.CacheHit();
  stat[1] = schedules.CacheMiss();
  stat[2] = schedules.LoopTripCount(1);
  stat[3] = schedules.LoopTripCount(2);
  stat[4] = schedules.LoopTripCount(3);
}

extern "C" {

void TemporalCse(const int64_t* rattr_ptr, const int64_t* aattr_ptr,
                 uint64_t n, uint64_t* cost, char* brepr,
                 uint64_t* operations, uint64_t* stat) {
  int64_t max_rattr = rattr_ptr[n - 1];
  if (max_rattr < numeric_limits<int8_t>::max()) {
    if (n < numeric_limits<int8_t>::max()) {
      TemporalCseKernel<int8_t, int8_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else if (n < numeric_limits<int16_t>::max()) {
      TemporalCseKernel<int8_t, int16_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else if (n < numeric_limits<int32_t>::max()) {
      TemporalCseKernel<int8_t, int32_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else {
      TemporalCseKernel<int8_t, int64_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    }
  } else if (max_rattr < numeric_limits<int16_t>::max()) {
    if (n < numeric_limits<int8_t>::max()) {
      TemporalCseKernel<int16_t, int8_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else if (n < numeric_limits<int16_t>::max()) {
      TemporalCseKernel<int16_t, int16_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else if (n < numeric_limits<int32_t>::max()) {
      TemporalCseKernel<int16_t, int32_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else {
      TemporalCseKernel<int16_t, int64_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    }
  } else if (max_rattr < numeric_limits<int32_t>::max()) {
    if (n < numeric_limits<int8_t>::max()) {
      TemporalCseKernel<int32_t, int8_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else if (n < numeric_limits<int16_t>::max()) {
      TemporalCseKernel<int32_t, int16_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else if (n < numeric_limits<int32_t>::max()) {
      TemporalCseKernel<int32_t, int32_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else {
      TemporalCseKernel<int32_t, int64_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    }
  } else {
    if (n < numeric_limits<int8_t>::max()) {
      TemporalCseKernel<int64_t, int8_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else if (n < numeric_limits<int16_t>::max()) {
      TemporalCseKernel<int64_t, int16_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else if (n < numeric_limits<int32_t>::max()) {
      TemporalCseKernel<int64_t, int32_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    } else {
      TemporalCseKernel<int64_t, int64_t>(
          rattr_ptr, aattr_ptr, n, cost, brepr, operations, stat);
    }
  }
}

}   // extern "C"
