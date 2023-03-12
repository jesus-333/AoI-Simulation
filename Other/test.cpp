#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
int main()
{
    float a = 1.132;
    vector<float> v(100);
    generate(v.begin(), v.end(), [n = 0, &a]() mutable { return n++ * a; });
}
