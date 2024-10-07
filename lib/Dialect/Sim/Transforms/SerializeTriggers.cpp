#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sim-serialize-triggers"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_SERIALIZETRIGGERS
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace sim;

namespace {

struct SerializeTriggersPass
    : public sim::impl::SerializeTriggersBase<SerializeTriggersPass> {

  void runOnOperation() override;
};

void SerializeTriggersPass::runOnOperation() {
  llvm::SmallSetVector<Value, 8> concurrentTriggers;

  auto theModule = getOperation();

  auto isConcurrent = [](Value trig) -> bool {
    return !trig.hasOneUse() && !trig.getUses().empty();
  };

  for (auto port : theModule.getPortList()) {
    if (port.isInput() && isa<EdgeTriggerType, InitTriggerType>(port.type)) {
      auto trigArg = theModule.getArgumentForPort(port.argNum);
      if (isConcurrent(trigArg))
        concurrentTriggers.insert(trigArg);
    }
  }

  theModule.walk([&](Operation *op) {
    if (isa<OnEdgeOp, OnInitOp, AnchoredTriggerOp, TriggerSequenceOp>(op))
      for (auto res : op->getResults())
        if (isConcurrent(res))
          concurrentTriggers.insert(res);
  });

  if (concurrentTriggers.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  for (auto trigger : concurrentTriggers) {
    OpBuilder builder(theModule);
    Location loc = theModule.getLoc();
    if (auto defOp = trigger.getDefiningOp()) {
      builder.setInsertionPointAfter(defOp);
      loc = defOp->getLoc();
    }
    auto numUses =
        std::distance(trigger.getUses().begin(), trigger.getUses().end());
    auto newSeq = builder.create<sim::TriggerSequenceOp>(loc, trigger, numUses);
    size_t resIdx = 0;
    for (auto &use : llvm::make_early_inc_range(trigger.getUses())) {
      if (use.getOwner() == newSeq)
        continue;
      use.assign(newSeq.getResult(resIdx));
      resIdx++;
    }
  }
}

} // namespace