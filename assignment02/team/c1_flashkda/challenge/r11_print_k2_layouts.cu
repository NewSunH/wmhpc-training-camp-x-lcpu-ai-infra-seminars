#include <cute/tensor.hpp>
#include "../FlashKDA/csrc/smxx/fwd_kernel2.cuh"
using namespace cute;
int main() {
  using L = K2Layouts<128,16>;
  print("VO: "); print(typename L::VOLayout{}); print("\n");
  print("VO a: "); print(L::VOLayout{}.layout_a()); print("\n");
  print("VO b: "); print(L::VOLayout{}.layout_b()); print("\n");
  print("TMA VO: "); print(typename L::TMAVOLayout{}); print("\n");
  print("VS VO: "); print(typename L::ValueSliceVOLayout{}); print("\n");
  print("TMA VS VO: "); print(typename L::TMAValueSliceVOLayout{}); print("\n");
}
