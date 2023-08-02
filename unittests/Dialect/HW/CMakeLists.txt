add_circt_unittest(CIRCTHWTests
  APLogicTest.cpp
  HWModuleTest.cpp
  InstanceGraphTest.cpp
)

target_link_libraries(CIRCTHWTests
  PRIVATE
  CIRCTHW
)
