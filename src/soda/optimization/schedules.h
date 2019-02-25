#ifndef SCHEDULES_H_
#define SCHEDULES_H_

#include <algorithm>
#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if __cplusplus >= 201703L
#include <shared_mutex>
#endif

#include <glog/logging.h>

#include "brepr.h"
#include "schedule.h"

namespace tcse {
#if __cplusplus >= 201703L
  using Mutex= std::shared_mutex;
  using Lock = std::shared_lock<Mutex>;
#else
  using Mutex = std::mutex;
  using Lock = std::unique_lock<Mutex>;
#endif

enum ExplorationOrder : uint64_t {
  kNormal = 0,
  kReversed = 1,
  kRandom = 2,
  kFromMiddle = 3,
  kDefault = kFromMiddle
};

};  // namespace tcse

class ContextBase {
 public:
  using ReadLock = tcse::Lock;
  using WriteLock = std::unique_lock<tcse::Mutex>;

  ContextBase()
      : stat_(new std::array<uint64_t, 5>({})), stat_mtx_(new tcse::Mutex) {}

  void HitCache() {
    WriteLock lock(*stat_mtx_);
    (*stat_)[0] += 1;
  }
  void MissCache() {
    WriteLock lock(*stat_mtx_);
    (*stat_)[1] += 1;
  }
  void VisitLoop(int level) {
    WriteLock lock(*stat_mtx_);
    (*stat_)[level + 1] += 1;
  }
  uint64_t CacheHit() const {
    ReadLock lock(*stat_mtx_);
    return (*stat_)[0];
  }
  uint64_t CacheMiss() const {
    ReadLock lock(*stat_mtx_);
    return (*stat_)[1];
  }
  uint64_t LoopTripCount(int level) const {
    ReadLock lock(*stat_mtx_);
    return (*stat_)[level + 1];
  }
  void PrintStats(std::ostream&) const;
  template<typename T>
  void Export(T* stat) const {
    std::copy_n(stat_->begin(), stat_->size(), stat);
  }

 private:
  std::shared_ptr<std::array<uint64_t, 5>> stat_;
  std::shared_ptr<tcse::Mutex> stat_mtx_;
};

template<typename RAttr, typename AAttr>
class Schedules {
 public:
  // Alias
  using RAttrVec = std::vector<RAttr>;
  using AAttrVec = std::vector<AAttr>;
  using Schedule = templated::Schedule<RAttr, AAttr>;
  using ScheduleVec = std::vector<std::shared_ptr<const Schedule>>;
  using Operation = typename Schedule::Operation;
  using OperationSet = typename Schedule::OperationSet;
  using Operations = typename Schedule::Operations;
  using Slice = typename Schedule::Slice;
  using CacheType = std::map<std::pair<AAttr, AAttr>,
                             std::shared_ptr<const Schedules>>;
  using TasksType = std::map<std::pair<AAttr, AAttr>, std::atomic<bool>>;
  using ReadLock = ContextBase::ReadLock;
  using WriteLock = ContextBase::WriteLock;

  struct Context : public ContextBase {
    Context(const std::shared_ptr<CacheType>& cache)
        : ContextBase(), cache(cache), cache_mtx(new tcse::Mutex),
          tasks(new TasksType), tasks_mtx(new tcse::Mutex) {}

    std::vector<AAttr> Shuffle(AAttr n) const {
      std::vector<AAttr> vec(n);
      auto range_from_middle = [&n](AAttr i) {
        if (n % 2 == 0) {
          if (i % 2 == 0) { return n / 2 - i / 2 - 1; }
          return n / 2 + i / 2;
        }
        if (i == 0) { return n / 2; }
        if (i % 2 == 1) { return n / 2 - (i + 1) / 2; }
        return n / 2 + (i + 1) / 2;
      };
      switch (exploration_order) {
        case tcse::kNormal:
          for (AAttr i = 0; i < n; ++i) { vec[i] = i; }
          return vec;
        case tcse::kReversed:
          for (AAttr i = 0; i < n; ++i) { vec[i] = n - 1 - i; }
          return vec;
        case tcse::kRandom:
          for (AAttr i = 0; i < n; ++i) { vec[i] = i; }
          std::shuffle(vec.begin(), vec.end(),
                       std::default_random_engine(seed));
          return vec;
        case tcse::kFromMiddle:
          for (AAttr i = 0; i < n; ++i) { vec[i] = range_from_middle(i); }
          return vec;
        default: return vec;
      }
    }

    std::shared_ptr<CacheType> Cache() const { return cache.lock(); }

    std::weak_ptr<CacheType> cache;
    std::shared_ptr<tcse::Mutex> cache_mtx;
    std::shared_ptr<TasksType> tasks;
    std::shared_ptr<tcse::Mutex> tasks_mtx;
    static tcse::ExplorationOrder exploration_order;
    static unsigned seed;
  };

  // Constructors
  Schedules(const std::shared_ptr<CacheType>& cache,
            const std::vector<RAttr>& rattr,
            const std::vector<AAttr>* aattr = nullptr)
      : Schedules(Context(cache), rattr, aattr) {}
  Schedules(const Context& context,
            const std::vector<RAttr>& rattr,
            const std::vector<AAttr>* aattr = nullptr,
            AAttr num_ops = -1, AAttr offset = 0,
            AAttr max_cost = -1);

  // Other functions
  Operation MakeOperation(const Slice&, const BRepr&) const;
  const ScheduleVec& GetSchedules(AAttr num_ops, AAttr offset);
  const ScheduleVec& Generate();
  Schedule Best();
  template<typename T>
  void Export(T* stat) const { context_.Export(stat); }

 private:
  std::shared_ptr<const Schedules> GetCache(AAttr num_ops, AAttr offset) {
    std::shared_ptr<const Schedules> schedules;
    if (auto cache = context_.Cache()) {
      const auto key = std::make_pair(num_ops, offset);
      ReadLock lock(*context_.cache_mtx);
      if (cache->count(key)) {
        context_.HitCache();
        schedules = (*cache)[key];
      } else {
        context_.MissCache();
      }
    }
    return schedules;
  }
  void SetCache(AAttr num_ops, AAttr offset,
                const std::shared_ptr<const Schedules>& schedules) {
    if (auto cache = context_.Cache()) {
      WriteLock lock(*context_.cache_mtx);
      (*cache)[std::make_pair(num_ops, offset)] = schedules;
    }
  }

  Context context_;
  const RAttrVec* rattr_;
  const AAttrVec* aattr_;
  AAttr num_ops_;
  AAttr offset_;
  AAttr max_cost_;
  ScheduleVec schedules_;
};

template<typename RAttr, typename AAttr>
tcse::ExplorationOrder Schedules<RAttr, AAttr>::Context::exploration_order =
    tcse::kDefault;

template<typename RAttr, typename AAttr>
unsigned Schedules<RAttr, AAttr>::Context::seed = 42;
#endif  // SCHEDULES_H_
