#ifndef SCHEDULE_H_
#define SCHEDULE_H_

#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "brepr.h"

namespace templated {

template<typename RAttr, typename AAttr>
struct Schedule {
  using RAttrVec = std::vector<RAttr>;
  using AAttrVec = std::vector<AAttr>;
  using Slice = std::pair<AAttr, AAttr>;
  struct Operation {
    struct Hasher {
      size_t operator()(const Operation& key) const {
        size_t result = 0x345678L;
        size_t mult = 1000003L;
        for (const auto& rattr : key.rattrs) {
          result = result ^ std::hash<RAttr>()(rattr) * mult;
          mult += 82520L + key.rattrs.size() + key.rattrs.size();
        }
        result ^= 97531L;
        return result ^ std::hash<BRepr>()(*key.brepr);
      }
    };

    bool operator==(const Operation& rhs) const {
      if (*brepr != *rhs.brepr) { return false; }
      if (rattrs != rhs.rattrs) { return false; }
      if (aattr == nullptr && rhs.aattr == nullptr) { return true; }
      if (aattr == nullptr || rhs.aattr == nullptr) { return false; }
      for (size_t i = 0; i < rattrs.size(); ++i) {
        if (aattr[i] != rhs.aattr[i]) { return false; }
      }
      return true;
    }

    RAttrVec rattrs;
    const AAttr* aattr;
    const BRepr* brepr;
  };
  using Operations = std::vector<Slice>;
  using OperationSet = std::unordered_set<Operation,
                                          typename Operation::Hasher>;

  Schedule()
      : brepr(), operations(), operation_set(), rattr(nullptr), aattr(nullptr),
        cost(std::numeric_limits<AAttr>::max()) {}
  Schedule(const std::shared_ptr<const BRepr>& brepr,
           const std::shared_ptr<const Operations>& operations,
           const std::shared_ptr<const OperationSet>& operation_set,
           const RAttrVec* rattr, const AAttrVec* aattr)
      : brepr(brepr), operations(operations), operation_set(operation_set),
        rattr(rattr), aattr(aattr), cost(operation_set->size() + 1) {}

  std::shared_ptr<const BRepr> brepr;
  std::shared_ptr<const Operations> operations;
  std::shared_ptr<const OperationSet> operation_set;
  const RAttrVec* rattr;
  const AAttrVec* aattr;
  AAttr cost;
};

}   // namespace templated

#endif  // SCHEDULE_H_
