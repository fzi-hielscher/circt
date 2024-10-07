add_circt_conversion_library(CIRCTPrintsToArcEnvCalls
  PrintsToArcEnvCalls.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTSim
  MLIRLLVMCommonConversion
  MLIRTransforms
)
