#ifndef SCHEDULES_H_
#define SCHEDULES_H_

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "brepr.h"
#include "schedule.h"

class ScheduleBase {
 public:
  using StatType = std::array<uint64_t, 5>;

  ScheduleBase(const std::shared_ptr<StatType>& stat) : stat_(stat) {}

  void HitCache() { if (stat_) { (*stat_)[0] += 1; } }
  void MissCache() { if (stat_) { (*stat_)[1] += 1; } }
  void VisitLoop(int level) { if (stat_) { (*stat_)[level + 1] += 1; } }
  uint64_t CacheHit() { return (*stat_)[0]; }
  uint64_t CacheMiss() { return (*stat_)[1]; }
  uint64_t LoopTripCount(int level) { return (*stat_)[level + 1]; }
  void PrintStats(std::ostream&);

 protected:
  std::shared_ptr<StatType> stat_;
};

template<typename RAttr, typename AAttr>
class Schedules : public ScheduleBase {
 public:
  // Alias
  using RAttrVec = std::vector<RAttr>;
  using AAttrVec = std::vector<AAttr>;
  using CacheType = std::unordered_map<
      AAttr, std::unordered_map<AAttr, std::shared_ptr<Schedules>>>;
  using Schedule = templated::Schedule<RAttr, AAttr>;
  using ScheduleVec = std::vector<std::shared_ptr<const Schedule>>;
  using Operation = typename Schedule::Operation;
  using OperationSet = typename Schedule::OperationSet;
  using Operations = typename Schedule::Operations;
  using Slice = typename Schedule::Slice;

  // Constructors
  Schedules(const std::vector<RAttr>& rattr,
            const std::vector<AAttr>* aattr = nullptr,
            std::shared_ptr<CacheType> cache = nullptr,
            AAttr num_ops = -1, AAttr offset = 0,
            std::shared_ptr<StatType> stat = nullptr, AAttr max_cost = -1);

  // Other functions
  Operation MakeOperation(const Slice&, const BRepr&);
  const ScheduleVec& GetSchedules(AAttr num_ops, AAttr offset);
  const ScheduleVec& Generate();
  Schedule Best();

 private:
  std::shared_ptr<CacheType> Cache() { return cache_.lock(); }

  const RAttrVec* rattr_;
  const AAttrVec* aattr_;
  std::weak_ptr<CacheType> cache_;
  AAttr num_ops_;
  AAttr offset_;
  AAttr max_cost_;
  ScheduleVec schedules_;
};

#endif  // SCHEDULES_H_
