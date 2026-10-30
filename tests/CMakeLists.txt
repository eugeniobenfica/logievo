add_executable(logievo_core_tests CoreTests.cpp)
target_link_libraries(logievo_core_tests PRIVATE logievo_core)
add_test(NAME core COMMAND logievo_core_tests)

# The canvas driven with synthetic mouse and keys (offscreen: no window shows).
find_package(Qt6 REQUIRED COMPONENTS Test)
add_executable(logievo_ui_tests UiTests.cpp)
target_link_libraries(logievo_ui_tests PRIVATE logievo_ui Qt6::Test)
add_test(NAME ui COMMAND logievo_ui_tests)
set_tests_properties(ui PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
