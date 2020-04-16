#include <stdio.h>
#include <iostream>
#include<algorithm>

int main() {
int cnt, mnt;
char str[256] = "cnt = 56,78,vertex=333";
char str2[256];
int a = sscanf(str, "cnt = %d,%d,%s", &cnt, &mnt, str2);
std::cout << cnt << '\n' << mnt << '\n' << str2 << '\n' << a << std::endl;
std::string aa = "/mnt/mfs/yiling/records/Deepfake/face2face/3dmm/000/000_0389_0.obj";
aa.replace(52, 3, "000_003");
aa.replace(47, 4, "swap");
std::cout << aa << std::endl;
return 0;
}
