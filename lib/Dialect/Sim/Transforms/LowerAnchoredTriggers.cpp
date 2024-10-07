#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sim-lower-anchored-triggers"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_LOWERANCHOREDTRIGGERS
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace sim;

namespace {

struct LowerAnchoredTriggersPass
    : public sim::impl::LowerAnchoredTriggersBase<LowerAnchoredTriggersPass> {

  void
  anchoredTriggersToSequences(hw::HWModuleOp moduleOp,
                              ArrayRef<AnchoredTriggerOp> anchoredTriggers);
  void runOnOperation() override;
};

void LowerAnchoredTriggersPass::anchoredTriggersToSequences(
    hw::HWModuleOp moduleOp, ArrayRef<AnchoredTriggerOp> anchoredTriggers) {
  SmallDenseMap<Value, SmallVector<AnchoredTriggerOp>> triggerGroups;

  auto rootTrigger = [](AnchoredTriggerOp anchor) -> Value {
    while (true) {
      if (auto parentAnchor =
              anchor.getParent().getDefiningOp<AnchoredTriggerOp>())
        anchor = parentAnchor;
      else
        break;
    }
    return anchor.getParent();
  };

  for (auto ato : anchoredTriggers) {
    auto root = rootTrigger(ato);
    triggerGroups[root].emplace_back(ato);
  }

  OpBuilder builder(moduleOp);
  for (auto &[root, leafs] : triggerGroups) {
    assert(!leafs.empty());

    if (leafs.size() == 1) {
      leafs.front().replaceAllUsesWith(root);
      leafs.front().erase();
      continue;
    }

    builder.setInsertionPointAfterValue(root);
    SmallVector<Location> locs;
    locs.reserve(leafs.size());
    for (auto leaf : leafs)
      locs.push_back(leaf.getLoc());
    auto fusedLoc = FusedLoc::get(builder.getContext(), locs);
    auto seqOp =
        builder.create<TriggerSequenceOp>(fusedLoc, root, leafs.size());
    for (auto [leaf, result] : llvm::zip(leafs, seqOp.getResults())) {
      leaf.replaceAllUsesWith(result);
      leaf.erase();
    }
  }
}

void LowerAnchoredTriggersPass::runOnOperation() {

  SmallVector<AnchoredTriggerOp> anchoredOps;
  for (auto op : getOperation().getOps<AnchoredTriggerOp>())
    anchoredOps.push_back(op);

  if (anchoredOps.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  anchoredTriggersToSequences(getOperation(), anchoredOps);
}

} // namespace