add_circt_dialect_library(CIRCTSimTransforms
  LowerPrintsToSV.cpp
  ProceduralizeSim.cpp


  DEPENDS
  CIRCTSimTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSim
  CIRCTSeq
  CIRCTSV
  CIRCTComb
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
)
