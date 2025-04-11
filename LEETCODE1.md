# 中位数

## 4.寻找两个正序数组的中位数（裁剪）

[ 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数** 。

算法的时间复杂度应该为 `O(log (m+n))` 。

```cpp
class Solution {
public:
    //裁剪法
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int totallength=nums1.size()+nums2.size();
        if(totallength%2==1){
            return getKthElement(nums1,nums2,(totallength+1)/2);
        }else{
            return (getKthElement(nums1,nums2,totallength/2)+getKthElement(nums1,nums2,totallength/2+1))/2.0;
        }
    }
private:
    int getKthElement(const vector<int>&nums1,const vector<int>& nums2,int k){
        int m=nums1.size();
        int n=nums2.size();
        int index1=0,index2=0;

        while(true){
            //边界情况
            if(index1==m){
                return nums2[index2+k-1];
            }
            if(index2==n){
                return nums1[index1+k-1];
            }
            if(k==1){
                return min(nums1[index1],nums2[index2]);
            }
            //正常情况
            //每轮去掉最不能存在中位数的序列，然后更新为更小的序列问题
            int newindex1=min(index1+k/2-1,m-1);
            int newindex2=min(index2+k/2-1,n-1);
            int pivot1=nums1[newindex1];
            int pivot2=nums2[newindex2];
            if(pivot1<=pivot2){  //去掉nums1左半边
                k-=newindex1-index1+1;
                index1=newindex1+1;
            }else{  //去掉nums右半边
                k-=newindex2-index2+1;
                index2=newindex2+1;
            }
        }
    }
};
```



# 接雨水

## 11.盛最多水的容器

[11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：**你不能倾斜容器。

![img](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)



```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i=0,j=height.size()-1,res=0;
        while(i<j){
            res=height[i]<height[j]?
            max(res,(j-i)*height[i++]):
            max(res,(j-i)*height[j--]);
        }
        return res;
    }
};
```

## 42.接雨水

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png)

```cpp
class Solution {
public:
    //单调栈法：
    //单调栈存储的是下标 从栈底到栈顶的下标对应的数组元素(递减)
    int trap(vector<int>& height) {
        int ans=0;
        stack<int> stk;
        int n=height.size();
        for(int i=0;i<n;i++){
            //记单调栈中 栈顶元素为top top的下一个元素为left
            //如果height[i]>height[top],得到可以接雨水的区域
            //宽度为 i-left-1 高度为 min(height[left],height[i])-height[top];
            while(!stk.empty()&&height[i]>height[stk.top()]){
                int top=stk.top();
                stk.pop();//当前最小高度
                if(stk.empty()) break;
                int left=stk.top();
                int currWidth=i-left-1;
                int currHeight=min(height[left],height[i])-height[top];
                ans+=currWidth*currHeight;
            }
            //若栈内元素均小于height[i],则到此处时栈为空
            //否则站内元素均大于 height[i]
            //即将height[i]压入栈时，栈仍然保持单调
            stk.push(i);//压入当前高度
        }
        return ans;
    }
};
```



```cpp
class Solution {
public:
    //动态规划
    //对于下标i，下雨后水能够达到的最大高度=下标i min(左边高度最大值，右边高度最大值）
    //下标i处能接的雨水量=i处下雨后水能够达到的最大高度-height[i]
    int trap(vector<int>& height) {
        int n=height.size();
        if(n==0) return 0;
        //leftMax[i] 保存i 左边(包含i)的最大高度
        vector<int> leftMax(n);
        leftMax[0]=height[0];
        for(int i=1;i<n;++i){
            leftMax[i]=max(leftMax[i-1],height[i]);
        }
        //rightMax[i] 保存i 右边(包含i)的最大高度
        vector<int> rightMax(n);
        rightMax[n-1]=height[n-1];
        for(int i=n-2;i>=0;--i){
            rightMax[i]=max(rightMax[i+1],height[i]);
        }

        int ans=0;
        for(int i=0;i<n;++i){
            ans+=min(leftMax[i],rightMax[i])-height[i];
        }
        return ans;
    
```



## 84.柱状图中最大的矩形(单调栈)

[(https://leetcode.cn/problems/largest-rectangle-in-histogram/)

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

**示例 1:**

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg)

```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram-1.jpg)

```
输入： heights = [2,4]
输出： 4
```

 

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int n = heights.size();
        vector<int> left(n), right(n);
        // 2 1 5 6 2 3
        stack<int> mono_stack;
        for (int i = 0; i < n; ++i) {  //栈从左往右栈保持递增
            while (!mono_stack.empty() && heights[mono_stack.top()] >= heights[i]) {
                mono_stack.pop();
            }
            //LEFT 为当前值左侧保持栈递增的首个值
            left[i] = (mono_stack.empty() ? -1 : mono_stack.top());
            mono_stack.push(i);
        }
		// i = 0  EMPTY LEFT[0] = -1
        // i = 1  STACK {0} POP{0} STACK {1} EMPTY LEFT[1] = -1
        // i = 2  STACK {1} LEFT[2] = 1 STACK {1,2} LEFT[2] = 1
        // i = 3  STACK {1,2} STACK {1,2,3} LEFT[3] = 2
        // i = 4  STACK {1,2,3} POP{2,3}  LEFT[4] = 2
        // i = 5  STACK {1,4} STACK{1,4,5} LEFT[5] = 4
        
        mono_stack = stack<int>();
        for (int i = n - 1; i >= 0; --i) {  //栈从右往左保持递减
            while (!mono_stack.empty() && heights[mono_stack.top()] >= heights[i]) {
                mono_stack.pop();
            }
            //Right 为当前值右侧保持栈递减的首个值
            right[i] = (mono_stack.empty() ? n : mono_stack.top());
            mono_stack.push(i);
        }
        
        // i = 5  EMPTY RIGHT[5] = 5
        // i = 4  STACK {5} POP {5} STACK{4} RIGHT[4] = 5
        // i = 3  STACK {4} STACK{4,3} RIGHT[3] = 4
        // i = 2  STACK {4,3} POP{4} STACK{2,3} RIGHT[2] = 4
        // i = 1  STACK {2,3} POP{2,3} STACK{1} RIGHT[1] = 2
        // i = 0  STACK {1} STACK{0,1} RIGHT[0] = 1
        
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans = max(ans, (right[i] - left[i] - 1) * heights[i]);
        }
        return ans;
    }
};
```



## 85.最大矩形

[(https://leetcode.cn/problems/maximal-rectangle/)

给定一个仅包含 `0` 和 `1` 、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

![img](https://pic.leetcode.cn/1722912576-boIxpm-image.png)

```
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6
解释：最大矩形如上图所示。
```

**示例 2：**

```
输入：matrix = [["0"]]
输出：0
```

**示例 3：**

```
输入：matrix = [["1"]]
输出：1
```

![fig2](https://assets.leetcode-cn.com/solution-static/85/3_1.png)

#### 方法一: 使用柱状图的优化暴力方法

```cpp
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if (m == 0) {
            return 0;
        }
        int n = matrix[0].size();
        vector<vector<int>> left(m, vector<int>(n, 0));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    left[i][j] = (j == 0 ? 0: left[i][j - 1]) + 1;
                }
            }
        }
        
        //计算每个点的最大宽度

        int ret = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '0') {
                    continue;
                }
                int width = left[i][j];
                int area = width;
                for (int k = i - 1; k >= 0; k--) {  //检查从上往下的最小宽度
                    width = min(width, left[k][j]);
                    area = max(area, (i - k + 1) * width);
                }
                ret = max(ret, area);
            }
        }
        return ret;
    }
};
```



#### 方法二：单调栈

```cpp
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if (m == 0) {
            return 0;
        }
        int n = matrix[0].size();
        vector<vector<int>> left(m, vector<int>(n, 0));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    left[i][j] = (j == 0 ? 0: left[i][j - 1]) + 1;
                }
            }
        }

        int ret = 0;
        for (int j = 0; j < n; j++) { // 对于每一列，使用基于柱状图的方法
            vector<int> up(m, 0), down(m, 0);

            stack<int> stk;
            for (int i = 0; i < m; i++) {
                while (!stk.empty() && left[stk.top()][j] >= left[i][j]) {
                    stk.pop();
                }
                up[i] = stk.empty() ? -1 : stk.top();
                stk.push(i);
            }
            stk = stack<int>();
            for (int i = m - 1; i >= 0; i--) {
                while (!stk.empty() && left[stk.top()][j] >= left[i][j]) {
                    stk.pop();
                }
                down[i] = stk.empty() ? m : stk.top();
                stk.push(i);
            }

            for (int i = 0; i < m; i++) {
                int height = down[i] - up[i] - 1;
                int area = height * left[i][j];
                ret = max(ret, area);
            }
        }
        return ret;
    }
};
```



# K数之和

## 15.三数之和（双指针）

[15. 三数之和](https://leetcode.cn/problems/3sum/)

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请你返回所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        if(nums.size()<3)  return ans;
        sort(nums.begin(),nums.end());
        //k为最小 i为次小 j为最大
        for(int k=0;k<nums.size()-2;k++){
            if(nums[k]>0) break;
            if(k>0&&nums[k]==nums[k-1]) continue;
            int i=k+1,j=nums.size()-1;
            while(i<j){
                int sum=nums[k]+nums[i]+nums[j];
                if(sum<0){
                    while(i<j&&nums[i]==nums[++i]);
                }else if(sum>0){
                    while(i<j&&nums[j]==nums[--j]);
                }else{//sum==0
                    vector<int> t;
                    t.push_back(nums[k]);
                    t.push_back(nums[i]);
                    t.push_back(nums[j]);
                    ans.push_back(t);
                    while(i<j&&nums[i]==nums[++i]);
                    while(i<j&&nums[j]==nums[--j]);
                }
            }
        }
        return ans;
    }
};
```

## 16.最接近的三数之和（双指针）

[16. 最接近的三数之和](https://leetcode.cn/problems/3sum-closest/)

给你一个长度为 `n` 的整数数组 `nums` 和 一个目标值 `target`。请你从 `nums` 中选出三个整数，使它们的和与 `target` 最接近。

返回这三个数的和。

假定每组输入只存在恰好一个解。

```cpp
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        //排序+双指针
        sort(nums.begin(),nums.end());
        int n=nums.size();
        int mindiff=INT_MAX,ans=0;
        for(int i=0;i<n;i++){
            for(int j=i+1,k=n-1;j<k;){
                int temp=nums[i]+nums[j]+nums[k];
                if(abs(temp-target)<mindiff){
                    mindiff=abs(temp-target);
                    ans=temp;
                }
                if(temp>target){k--;}
                else if(temp<target){j++;}
                else return ans;
            }
        }
        return ans;
    }
};
```

## 18.四数之和（剪枝+双指针）

[18. 四数之和](https://leetcode.cn/problems/4sum/)

给你一个由 `n` 个整数组成的数组 `nums` ，和一个目标值 `target` 。请你找出并返回满足下述全部条件且**不重复**的四元组 `[nums[a], nums[b], nums[c], nums[d]]` （若两个四元组元素一一对应，则认为两个四元组重复）：

- `0 <= a, b, c, d < n`
- `a`、`b`、`c` 和 `d` **互不相同**
- `nums[a] + nums[b] + nums[c] + nums[d] == target`

你可以按 **任意顺序** 返回答案 。

```cpp
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> q;
        if(nums.size()<4) return q;
        sort(nums.begin(),nums.end());
        int n=nums.size();
        for(int i=0;i<n-3;i++){
            //去重复 剪枝1 2
            if(i>0&&nums[i]==nums[i-1]) continue;
            if((long)nums[i]+nums[i+1]+nums[i+2]+nums[i+3]>target) break;
            if((long)nums[i]+nums[n-3]+nums[n-2]+nums[n-1]<target) continue;
            for(int j=i+1;j<n-2;j++){
                //去重复 剪枝
                if(j>i+1&&nums[j]==nums[j-1]) continue;
                if((long)nums[i]+nums[j]+nums[j+1]+nums[j+2]>target) break;
                if((long)nums[i]+nums[j]+nums[n-2]+nums[n-1]<target) continue;
                int l=j+1,r=n-1;
                while(l<r){
                    int sum=nums[i]+nums[j]+nums[l]+nums[r];
                    if(sum==target){
                        q.push_back({nums[i],nums[j],nums[l],nums[r]});
                        while(l<r&&nums[l]==nums[l+1]){l++;} l++; 
                        while(l<r&&nums[r]==nums[r-1]){r--;} r--; 
                    }else if(sum<target){l++;}
                    else{ r--; }
                }
            }
        }
        return q;
    }
};

//如何避免枚举到重复的四元组？
//(为了避免枚举到重复的四元组，则需要保证每一重循环枚举到的元素不小于其
//上一重循环枚举到的元素，且在同一个循环中不能多次枚举到相同元素)

//----->排序+双指针:
//使用排序+双指针 可以减少一重循环
//使用两重循环分别枚举前两个数，然后使用双指针枚举剩下两个数
//假设两重循环枚举到的前两个数分别位于下标i,j
//左右指针分别指向下标j+1和下标n-1每次计算4个数的和


//剪枝优化：
//1 确定第一个数后，如果nums[i]+nums[i+1]+num[i+2]+nums[i+3]>target
// 则剩下的三个数无论取什么值，四数之和一定大于target
//2 确定第第一个数后，如果nums[i]+nums[n-3]+nums[n-2]+nums[n-1]<target
// 则剩下的三个数无论取什么值，四数之和一定小于target
//3 确定前两个数之后，如果nums[i]+nums[j]+nums[n-2]+nums[n-1]<target
// 说明此时剩下的两个数无论取什么值，四数之和一定小于target
```

# 移除元素与查找

## 26.删除有序数组中的重复项

[26. 删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

给你一个 **非严格递增排列** 的数组 `nums` ，请你**[ 原地](http://baike.baidu.com/item/原地算法)** 删除重复出现的元素，使每个元素 **只出现一次** ，返回删除后数组的新长度。元素的 **相对顺序** 应该保持 **一致** 。然后返回 `nums` 中唯一元素的个数。

考虑 `nums` 的唯一元素的数量为 `k` ，你需要做以下事情确保你的题解可以被通过：

- 更改数组 `nums` ，使 `nums` 的前 `k` 个元素包含唯一元素，并按照它们最初在 `nums` 中出现的顺序排列。`nums` 的其余元素与 `nums` 的大小不重要。
- 返回 `k` 。

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n=nums.size();
        if(n==0) return 0;
        int k=0;
        for(int i=0;i<n;i++){
            if(nums[k]!=nums[i]){
                nums[++k]=nums[i];
            }
        }
        return k+1;
    }
};
```

## 27.移除元素

[27. 移除元素](https://leetcode.cn/problems/remove-element/)

给你一个数组 `nums` 和一个值 `val`，你需要 **[原地](https://baike.baidu.com/item/原地算法)** 移除所有数值等于 `val` 的元素。元素的顺序可能发生改变。然后返回 `nums` 中与 `val` 不同的元素的数量。

假设 `nums` 中不等于 `val` 的元素数量为 `k`，要通过此题，您需要执行以下操作：

- 更改 `nums` 数组，使 `nums` 的前 `k` 个元素包含不等于 `val` 的元素。`nums` 的其余元素和 `nums` 的大小并不重要。
- 返回 `k`。

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int k=0,n=nums.size();
        for(int i=0;i<n;i++){
            if(nums[i]==val) {k++; continue;}
            nums[i-k]=nums[i];
        }
        return n-k;
    }
};
```



## 34.在排序数组中查找元素的第一个和最后一个位置（二分法）

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

你必须设计并实现时间复杂度为 `O(log n)` 的算法解决此问题。

```cpp
class Solution {
public:
    //第一个等于target的位置--- leftIdx
    //第一个大于target的位置-1 ---- rightIdx
    int binarySearch(vector<int>& nums,int target,bool lower){
        int left=0,right=(int)nums.size()-1,ans=(int)nums.size();
        while(left<=right){
            int mid=(left+right)/2;
            //leftIdx---lower为true nums[mid]>=target 
            //             (mid最终指向第一个等于target位置)
            //leftIdx---lower为false nums[mid]>target
            //             (mid最终指向第一个大于target位置)
            if(nums[mid]>target||(lower&&nums[mid]>=target)){
                right=mid-1;
                ans=mid;
            }else{
                left=mid+1;
            }
        }
        return ans;
    }

    vector<int> searchRange(vector<int>& nums, int target) {
       int leftIdx=binarySearch(nums,target,true);
       int rightIdx=binarySearch(nums,target,false)-1;
       if(leftIdx<=rightIdx&&rightIdx<nums.size()
        &&nums[rightIdx]==target&&nums[leftIdx]==target){
            return vector<int> {leftIdx,rightIdx};
        }
        return vector<int> {-1,-1};
    }
};
```

## 41.缺失的第一个正数

[(https://leetcode.cn/problems/first-missing-positive/)

给你一个未排序的整数数组 `nums` ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 `O(n)` 并且只使用常数级别额外空间的解决方案。（类比冒泡排序）

```cpp
//置换
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n=nums.size();
        for(int i=0;i<n;i++){
            while(nums[i]>0&&nums[i]<=n&&nums[nums[i]-1]!=nums[i]){
                swap(nums[nums[i]-1],nums[i]);
            }
        }
        for(int i=0;i<n;i++){
            if(nums[i]!=i+1) return i+1;
        }
        return n+1;
    }
};
```

## 49.字母异位词分组 (hash 表)

[(https://leetcode.cn/problems/group-anagrams/)

给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的所有字母得到的一个新单词。

 

**示例 1:**

```
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

```cpp
class Solution {
public:
    //互为字母异位词的两个字符包含的字母相同，因此两个字符串中的相同字符
    //出现的次数一定是相同的，可以使用每个字母出现的次数使用字符串表示
    //作为哈希表的键
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        //自定义对array<int,26> 类型的哈希函数
        //这里
        auto arrayHash=[fn=hash<int>{}](const array<int,26>& arr)->size_t{
            return accumulate(arr.begin(),arr.end(),0u,[&](size_t acc,int num){
                return (acc<<1)^fn(num);
            });
        };

        unordered_map<array<int,26>,vector<string>,decltype(arrayHash)> mp(0,arrayHash);
        for(string& str:strs){
            array<int,26> counts{};
            int length=str.length();
            //统计对应字母出现的个数
            for(int i=0;i<length;++i){
                counts[str[i]-'a']++;
            }
            mp[counts].emplace_back(str);
        }
        vector<vector<string>> ans;
        for(auto it=mp.begin();it!=mp.end();it++){
            ans.emplace_back(it->second);
        }
        return ans;
    }
};
```



------

### **1. 自定义哈希函数 `arrayHash`**

```cpp
auto arrayHash = [fn = hash<int>{}](const array<int, 26>& arr) -> size_t {
    return accumulate(arr.begin(), arr.end(), 0u, [&](size_t acc, int num) {
        return (acc << 1) ^ fn(num);
    });
};
```

#### **定义**

- `arrayHash` 是一个使用 **lambda表达式** 定义的自定义哈希函数，用于哈希一个 `std::array<int, 26>`。
- 使用 C++ 的标准库函数 `std::accumulate`，对数组中的所有元素进行迭代计算。

#### **逻辑**

- `hash<int>`：标准哈希函数，用于计算单个整数的哈希值。

- `arr`：一个包含 26 个整数的数组，传递给 `arrayHash`。

- ```
  std::accumulate
  ```

  ：

  - 参数：
    1. `arr.begin()` 和 `arr.end()`：表示遍历整个数组。
    2. `0u`：初始累积值，`0u` 表示无符号整数。
    3. 一个 lambda 表达式，定义累积的规则。

#### **内部的累积逻辑**

```cpp
[&](size_t acc, int num) {
    return (acc << 1) ^ fn(num);
}
```

- 对于数组中的每个元素 

  ```
  num
  ```

  ，将其与累积值 

  ```
  acc
  ```

   进行如下计算：

  1. `(acc << 1)`：将当前累积值 `acc` 左移一位（相当于乘以 2）。
  2. `fn(num)`：计算当前元素 `num` 的哈希值。
  3. `(acc << 1) ^ fn(num)`：将左移后的累积值与当前元素的哈希值通过异或运算组合起来，确保哈希值的分布更加均匀。

- 最终返回整个数组的哈希值。

------

### **2. 创建 `unordered_map`**

```cpp
unordered_map<array<int, 26>, vector<string>, decltype(arrayHash)> mp(0, arrayHash);
```

#### **定义**

- `unordered_map` 是一个哈希表，键是 `std::array<int, 26>` 类型，值是 `std::vector<std::string>` 类型。

- 第三个模板参数是 `decltype(arrayHash)`，用于指定自定义的哈希函数类型。

- 构造函数 

  ```
  mp(0, arrayHash)
  ```

  ：

  - 第一个参数 `0`：表示初始 bucket 数量。
  - 第二个参数 `arrayHash`：指定自定义的哈希函数。

#### **功能**

- `mp` 是一个无序哈希表，使用 `arrayHash` 来对键值 `array<int, 26>` 进行哈希计算，从而支持高效的查找和存储。



## 80.删除有序数组中的重复项 II（双指针）

[(https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/)

给你一个有序数组 `nums` ，请你**[ 原地](http://baike.baidu.com/item/原地算法)** 删除重复出现的元素，使得出现次数超过两次的元素**只出现两次** ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 **[原地 ](https://baike.baidu.com/item/原地算法)修改输入数组** 并在使用 O(1) 额外空间的条件下完成。

**示例 1：**

```
输入：nums = [1,1,1,2,2,3]
输出：5, nums = [1,1,2,2,3]
解释：函数应返回新长度 length = 5, 并且原数组的前五个元素被修改为 1, 1, 2, 2, 3。 不需要考虑数组中超出新长度后面的元素。
```

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n = nums.size();
        if (n <= 2) {
            return n;
        }
        int slow = 2, fast = 2;
        while (fast < n) {
            if (nums[slow - 2] != nums[fast]) {
                nums[slow] = nums[fast];
                ++slow;
            }
            ++fast;
        }
        return slow;
    }
};
```



## 81.搜索旋转排序数组 II（二分查找）

[(https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/)

已知存在一个按非降序排列的整数数组 `nums` ，数组中的值不必互不相同。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转** ，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,4,4,5,6,6,7]` 在下标 `5` 处经旋转后可能变为 `[4,5,6,6,7,0,1,2,4,4]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，请你编写一个函数来判断给定的目标值是否存在于数组中。

如果 `nums` 中存在这个目标值 `target` ，则返回 `true` ，否则返回 `false` 。

你必须尽可能减少整个操作步骤。

 

**示例 1：**

```
输入：nums = [2,5,6,0,0,1,2], target = 0
输出：true
```

**示例 2：**

```
输入：nums = [2,5,6,0,0,1,2], target = 3
输出：false
```



```cpp
class Solution {
public:
    bool search(vector<int> &nums, int target) {
        int n = nums.size();
        if (n == 0) {
            return false;
        }
        if (n == 1) {
            return nums[0] == target;
        }
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) {
                return true;
            }
            if (nums[l] == nums[mid] && nums[mid] == nums[r]) {
                ++l;
                --r;
            } else if (nums[l] <= nums[mid]) {
                if (nums[l] <= target && target < nums[mid]) { //在有序子序列中查找
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) { //在有序子序列中查找
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return false;
    }
};
```



# 排列

## 31.下一个排列

[31. 下一个排列](https://leetcode.cn/problems/next-permutation/)

整数数组的一个 **排列** 就是将其所有成员以序列或线性顺序排列。

- 例如，`arr = [1,2,3]` ，以下这些都可以视作 `arr` 的排列：`[1,2,3]`、`[1,3,2]`、`[3,1,2]`、`[2,3,1]` 。

整数数组的 **下一个排列** 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 **下一个排列** 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

- 例如，`arr = [1,2,3]` 的下一个排列是 `[1,3,2]` 。
- 类似地，`arr = [2,3,1]` 的下一个排列是 `[3,1,2]` 。
- 而 `arr = [3,2,1]` 的下一个排列是 `[1,2,3]` ，因为 `[3,2,1]` 不存在一个字典序更大的排列。

给你一个整数数组 `nums` ，找出 `nums` 的下一个排列。

必须**[ 原地 ](https://baike.baidu.com/item/原地算法)**修改，只允许使用额外常数空间。

```cpp
class Solution {
public:
    //  4 5 2 6 3 1  较小数i指向2 较大数j指向3
    //  4 5 3 6 2 1  交换后再重排
    //  4 5 3 1 2 6
    void nextPermutation(vector<int>& nums) {
        int i=nums.size()-2;
        //从后往前扫描  (降序)
        while(i>=0&&nums[i]>=nums[i+1]){ i--; }
        if(i>=0){
            int j=nums.size()-1;
            //找到在较小数
            while(j>=0&&nums[i]>=nums[j]){j--;}
            swap(nums[i],nums[j]);
        }
        reverse(nums.begin()+i+1,nums.end());
    }
};

//1 3 2 i指向1 j指向2
//2 3 1 交换后重排
//2 1 3
```

## 33.搜索旋转排序数组（二分法）

[搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

```cpp
class Solution {
public:
    // 4 5 6 7 0 1 2
    int search(vector<int>& nums, int target) {
        int n=(int)nums.size();
        if(!n) return -1;
        if(n==1) return nums[0]==target?0:-1;

        int l=0,r=n-1;
        while(l<=r){
            int mid=(l+r)/2;
            if(nums[mid]==target) return mid;
            if(nums[0]<=nums[mid]){
                //右有序
                if(nums[0]<=target&&target<nums[mid]){
                    r=mid-1;
                }else{  
                    l=mid+1;
                }
            }else{ //nums[0]>nums[mid]
                    //左有序
                if(nums[mid]<target&&target<=nums[n-1]){
                    l=mid+1;
                }else{
                    r=mid-1;
                }
            }
        }
        return -1;
    }
};
```





## 46.全排列(原数组无重复元素)

[(https://leetcode.cn/problems/permutations/)

给定一个不含重复数字的数组 `nums` ，返回其 *所有可能的全排列* 。你可以 **按任意顺序** 返回答案。

```cpp
class Solution {
public:
    void backtrack(vector<vector<int>>& res,vector<int>& output,int first,int len){
        //所有数都填完了
        if(first==len){
            res.emplace_back(output);
            return;
        }
        for(int i=first;i<len;++i){
            //动态维护数组
            //将first以后位置的数分别交换到当前位置
            swap(output[i],output[first]);
            //继续递归填充下一个位置
            backtrack(res,output,first+1,len);
            //撤销操作 复原位置
            swap(output[i],output[first]);
        }
    }

    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        backtrack(res,nums,0,(int)nums.size());
        return res;
    }
};
```



## 47.全排列 II（原数组包含重复元素）---> Visit 数组

[(https://leetcode.cn/problems/permutations-ii/)

给定一个可包含重复数字的序列 `nums` ，***按任意顺序*** 返回所有不重复的全排列。

```cpp
class Solution {
    vector<int> vis;
public:
    void backtrack(vector<int>& nums,vector<vector<int>>&ans,int idx,
    vector<int>& perm){
        if(idx==nums.size()){
            ans.emplace_back(perm); return;
        }
        for(int i=0;i<(int)nums.size();i++){
            //已填或出现重复元素
            if(vis[i]||(i>0&&nums[i]==nums[i-1]&&!vis[i-1])) continue;
            perm.emplace_back(nums[i]);
            vis[i]=1;
            backtrack(nums,ans,idx+1,perm);
            vis[i]=0;
            perm.pop_back();
        }
    }

    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> perm;
        vis.resize(nums.size());
        sort(nums.begin(),nums.end());
        backtrack(nums,ans,0,perm);
        return ans;
    }
};
```



## 78.子集

[(https://leetcode.cn/problems/subsets/)

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。

返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

000	{}	0
001	{9}	1
010	{2}	2
011	{2,9}	3
100	{5}	4
101	{5,9}	5
110	{5,2}	6
111	{5,2,9}	7



### A.幂集表示

```cpp
class Solution {
public:
    vector<int> t;
    vector<vector<int>> ans;

    vector<vector<int>> subsets(vector<int>& nums) {
        int n = nums.size();
        for (int mask = 0; mask < (1 << n); ++mask) {  // 2^n
            t.clear();
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    t.push_back(nums[i]);
                }
            }
            ans.push_back(t);
        }
        return ans;
    }
};
```



### B.递归法实现子集枚举

```cpp
class Solution {
public:
    vector<int> t;
    vector<vector<int>> ans;

    void dfs(int cur, vector<int>& nums) {
        if (cur == nums.size()) {
            ans.push_back(t);
            return;
        }
        t.push_back(nums[cur]);
        dfs(cur + 1, nums);  //选择当前值加入子集
        t.pop_back();
        dfs(cur + 1, nums);	  //不选择当前值加入子集
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(0, nums);
        return ans;
    }
};
```



# 矩阵、数独

## 36.有效的数独

(https://leetcode.cn/problems/valid-sudoku/)

请你判断一个 `9 x 9` 的数独是否有效。只需要 **根据以下规则** ，验证已经填入的数字是否有效即可。

1. 数字 `1-9` 在每一行只能出现一次。
2. 数字 `1-9` 在每一列只能出现一次。
3. 数字 `1-9` 在每一个以粗实线分隔的 `3x3` 宫内只能出现一次。（请参考示例图）

 

**注意：**

- 一个有效的数独（部分已被填充）不一定是可解的。
- 只需要根据以上规则，验证已经填入的数字是否有效即可。
- 空白格用 `'.'` 表示。

```cpp
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        int rows[9][9];
        int columns[9][9];
        int subboxes[3][3][9];

        memset(rows,0,sizeof(rows));
        memset(columns,0,sizeof(columns));
        memset(subboxes,0,sizeof(subboxes));
        for(int i=0;i<9;i++){
            for(int j=0;j<9;j++){
                char c=board[i][j];
                if(c!='.'){
                    int index=c-'0'-1;
                    rows[i][index]++;
                    columns[j][index]++;
                    subboxes[i/3][j/3][index]++;
                    if(rows[i][index]>1||columns[j][index]>1||subboxes[i/3][j/3][index]>1)
                    return false;
                }
            }
        }
        return true;
    }
};
```

## 37.解数独

[(https://leetcode.cn/problems/sudoku-solver/)

编写一个程序，通过填充空格来解决数独问题。

数独的解法需 **遵循如下规则**：

1. 数字 `1-9` 在每一行只能出现一次。
2. 数字 `1-9` 在每一列只能出现一次。
3. 数字 `1-9` 在每一个以粗实线分隔的 `3x3` 宫内只能出现一次。（请参考示例图）

数独部分空格内已填入了数字，空白格用 `'.'` 表示。

```cpp
class Solution {
private:
    bool line[9][9];
    bool column[9][9];
    bool block[3][3][9];
    bool valid;
    vector<pair<int,int>> spaces;
public:
    void dfs(vector<vector<char>>& board,int pos){
        if(pos==spaces.size()){
            valid=true;
            return;
        }
        auto [i,j]=spaces[pos];
        for(int digit=0;digit<9&&!valid;++digit){
            if(!line[i][digit]&&!column[j][digit]&&!block[i/3][j/3][digit]){
                line[i][digit]=column[j][digit]=block[i/3][j/3][digit]=true;
                //board 1-9
                board[i][j]=digit+'0'+1;
                dfs(board,pos+1);
                 line[i][digit]=column[j][digit]=block[i/3][j/3][digit]=false;
            }
        }
    }

    void solveSudoku(vector<vector<char>>& board) {
        memset(line,false,sizeof(line));
        memset(column,false,sizeof(column));
        memset(block,false,sizeof(block));
        valid=false;

        for(int i=0;i<9;++i){
            for(int j=0;j<9;++j){
                if(board[i][j]=='.'){
                    spaces.emplace_back(i,j);
                }else{
                    //board 1-9
                    int digit=board[i][j]-'0'-1;
                    line[i][digit]=column[j][digit]=block[i/3][j/3][digit]=true;
                }
            }
        }
        dfs(board,0);
    }
};
```



## 48.旋转图像

[(https://leetcode.cn/problems/rotate-image/)

给定一个 *n* × *n* 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在**[ 原地](https://baike.baidu.com/item/原地算法)** 旋转图像，这意味着你需要直接修改输入的二维矩阵。

**请不要** 使用另一个矩阵来旋转图像。

### A.使用辅助数组

```cpp
class Solution {
public:
    //n*n  [a,b] 顺时针旋转90 ----->[b,n-a+1]
    void rotate(vector<vector<int>>& matrix) {
        int n=matrix.size();
        //c++这里的=拷贝是值拷贝，会得到一个新的数组
        auto matrix_new=matrix;
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                matrix_new[j][n-i-1]=matrix[i][j];
            }
        }
        //这里也是值拷贝
        matrix=matrix_new;
    }
};
```

### B.原地旋转

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < (n + 1) / 2; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = temp;
            }
        }
    }
};
```



## 51.N 皇后（回溯）

[(https://leetcode.cn/problems/n-queens/)

按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

**n 皇后问题** 研究的是如何将 `n` 个皇后放置在 `n×n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n` ，返回所有不同的 **n 皇后问题** 的解决方案。

每一种解法包含一个不同的 **n 皇后问题** 的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

![img](https://assets.leetcode.com/uploads/2020/11/13/queens.jpg)

### 方法一：基于集合的回溯

```cpp
class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        auto solutions = vector<vector<string>>();
        auto queens = vector<int>(n, -1);
        auto columns = unordered_set<int>();
        auto diagonals1 = unordered_set<int>();
        auto diagonals2 = unordered_set<int>();
        backtrack(solutions, queens, n, 0, columns, diagonals1, diagonals2);
        return solutions;
    }

    void backtrack(vector<vector<string>> &solutions, vector<int> &queens, int n, int row, unordered_set<int> &columns, unordered_set<int> &diagonals1, unordered_set<int> &diagonals2) {
        if (row == n) {
            vector<string> board = generateBoard(queens, n);
            solutions.push_back(board);
        } else {
            for (int i = 0; i < n; i++) {
                if (columns.find(i) != columns.end()) {
                    continue;
                }
                int diagonal1 = row - i;
                if (diagonals1.find(diagonal1) != diagonals1.end()) {
                    continue;
                }
                int diagonal2 = row + i;
                if (diagonals2.find(diagonal2) != diagonals2.end()) {
                    continue;
                }
                queens[row] = i;
                columns.insert(i);
                diagonals1.insert(diagonal1);
                diagonals2.insert(diagonal2);
                backtrack(solutions, queens, n, row + 1, columns, diagonals1, diagonals2);
                queens[row] = -1;
                columns.erase(i);
                diagonals1.erase(diagonal1);
                diagonals2.erase(diagonal2);
            }
        }
    }

    vector<string> generateBoard(vector<int> &queens, int n) {
        auto board = vector<string>();
        for (int i = 0; i < n; i++) {
            string row = string(n, '.');
            row[queens[i]] = 'Q';
            board.push_back(row);
        }
        return board;
    }
};
```

## 54.螺旋矩阵

[(https://leetcode.cn/problems/spiral-matrix/)

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

![img](https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg)

```cpp
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

```cpp
class Solution {
private: 
    //分别是向右,向下,向左，向上
    static constexpr int directions[4][2]={{0,1},{1,0},{0,-1},{-1,0}};
public:
    //模拟法
    //基本思想：
    //初始位置是矩阵的左上角，初始方向是向右
    //当路径超出界限或进入之前访问的位置，顺时针旋转，进入下一个方向
    //------------------------------------------------------------
    //辅助矩阵visited：用于判断路径是否进入之前访问过的位置
    //------------------------------------------------------------
    //如何判断路径是否结束？
    //由于矩阵的每个元素都被访问一次，因此路径的长度即为矩阵中的元素数量
    //当路径长度达到矩阵中元素数量时即为完整路径，将该路径返回
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        //特殊情况
        if(matrix.size()==0||matrix[0].size()==0) return {};

        int rows=matrix.size(),columns=matrix[0].size();
        vector<vector<bool>> visited(rows,vector<bool>(columns));
        //初始化总路径长度
        int total=rows*columns; 
        //用于记录路径顺序
        vector<int> order(total);
        //从左上角出发，初始向右
        int row=0,column=0;
        int directionIndex=0;
        //开始模拟
        for(int i=0;i<total;i++){
            order[i]=matrix[row][column];
            visited[row][column]=true;
            //先检测
            int nextRow=row+directions[directionIndex][0];
            int nextColumn=column+directions[directionIndex][1];
            if(nextRow<0||nextRow>=rows||nextColumn<0||nextColumn>=columns
            ||visited[nextRow][nextColumn]){
                directionIndex=(directionIndex+1)%4;
            }
            //再更新
            row+=directions[directionIndex][0];
            column+=directions[directionIndex][1];
        }
        return order;
    }
};
```



## 59.螺旋矩阵 II

[(https://leetcode.cn/problems/spiral-matrix-ii/)

给你一个正整数 `n` ，生成一个包含 `1` 到 `n2` 所有元素，且元素按顺时针顺序螺旋排列的 `n x n` 正方形矩阵 `matrix` 。

### 方法一：模拟

```cpp
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        int maxNum = n * n;
        int curNum = 1;
        vector<vector<int>> matrix(n, vector<int>(n));
        int row = 0, column = 0;
        vector<vector<int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};  // 右下左上
        int directionIndex = 0;
        while (curNum <= maxNum) {
            matrix[row][column] = curNum;
            curNum++;
            int nextRow = row + directions[directionIndex][0];
            nextColumn = column + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= n || nextColumn < 0 
                || nextColumn >= n || matrix[nextRow][nextColumn] != 0) {
                directionIndex = (directionIndex + 1) % 4;  // 顺时针旋转至下一个方向
            }
            row = row + directions[directionIndex][0];
            column = column + directions[directionIndex][1];
        }
        return matrix;
    }
};
```



## 73.矩阵置零

[(https://leetcode.cn/problems/set-matrix-zeroes/)

给定一个 `*m* x *n*` 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用 **[原地](http://baike.baidu.com/item/原地算法)** 算法**。**

![img](https://assets.leetcode.com/uploads/2020/08/17/mat1.jpg)

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        int flag_col0 = false, flag_row0 = false;
        for (int i = 0; i < m; i++) {
            if (!matrix[i][0]) {
                flag_col0 = true;
            }
        }
        for (int j = 0; j < n; j++) {
            if (!matrix[0][j]) {
                flag_row0 = true;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (!matrix[i][j]) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (!matrix[i][0] || !matrix[0][j]) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (flag_col0) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
        if (flag_row0) {
            for (int j = 0; j < n; j++) {
                matrix[0][j] = 0;
            }
        }
    }
};
```



## 74.搜索二维矩阵(二分法)

[(https://leetcode.cn/problems/search-a-2d-matrix/)

给你一个满足下述两条属性的 `m x n` 整数矩阵：

- 每行中的整数从左到右按非严格递增顺序排列。
- 每行的第一个整数大于前一行的最后一个整数。

给你一个整数 `target` ，如果 `target` 在矩阵中，返回 `true` ；否则，返回 `false` 。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/10/05/mat.jpg)

```
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出：true
```

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int low = 0, high = m * n - 1;
        while (low <= high) {
            int mid = (high - low) / 2 + low;
            int x = matrix[mid / n][mid % n];
            if (x < target) {
                low = mid + 1;
            } else if (x > target) {
                high = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }
};
```

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>> matrix, int target) {
        auto row = upper_bound(matrix.begin(), matrix.end(), target, [](const int b, const vector<int> &a) {
            return b < a[0];
        });
        if (row == matrix.begin()) {
            return false;
        }
        --row;
        return binary_search(row->begin(), row->end(), target);
    }
};
```



## 79.单词搜索（回溯）

[(https://leetcode.cn/problems/word-search/)

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。

同一个单元格内的字母不允许被重复使用。

**例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```



```cpp
class Solution {
public:
    bool check(vector<vector<char>>& board, vector<vector<int>>& visited, int i, int j, string& s, int k) {
        if (board[i][j] != s[k]) {
            return false;
        } else if (k == s.length() - 1) {
            return true;
        }
        visited[i][j] = true;
        vector<pair<int, int>> directions{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        bool result = false;
        for (const auto& dir: directions) {
            int newi = i + dir.first, newj = j + dir.second;
            if (newi >= 0 && newi < board.size() && newj >= 0 && newj < board[0].size()) {
                if (!visited[newi][newj]) {
                    bool flag = check(board, visited, newi, newj, s, k + 1);
                    if (flag) {
                        result = true;
                        break;
                    }
                }
            }
        }
        visited[i][j] = false;
        return result;
    }

    bool exist(vector<vector<char>>& board, string word) {
        int h = board.size(), w = board[0].size();
        vector<vector<int>> visited(h, vector<int>(w));
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                bool flag = check(board, visited, i, j, word, 0);
                if (flag) {
                    return true;
                }
            }
        }
        return false;
    }
};
```



# 组合

## 39.组合总和

[(https://leetcode.cn/problems/combination-sum/)

给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target` ，

找出 `candidates` 中可以使数字和为目标数 `target` 的 所有 **不同组合** ，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。

`candidates` 中的 **同一个** 数字可以 **无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 `target` 的不同组合数少于 `150` 个。

```cpp
class Solution {
public:
    //回溯法
    void dfs(vector<int>& candidates,int target,vector<vector<int>>& ans,
    vector<int>& combine,int idx){
        if(idx==candidates.size()) return;
        if(target==0){
            ans.emplace_back(combine);
            return;
        } 
        //跳过当前数
        dfs(candidates,target,ans,combine,idx+1);
        //选择当前数
        if(target-candidates[idx]>=0){
            combine.emplace_back(candidates[idx]);
            dfs(candidates,target-candidates[idx],ans,combine,idx);
            combine.pop_back();  //删除combine的最后一个元素
        }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> ans;
        vector<int> combine;
        dfs(candidates,target,ans,combine,0);
        return ans;
    }
};
```



## 40.组合总和 II

[(https://leetcode.cn/problems/combination-sum-ii/)

给定一个候选人编号的集合 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。

`candidates` 中的每个数字在每个组合中只能使用 **一次** 。

**注意：**解集不能包含重复的组合。

```cpp
class Solution {
private:
    vector<int> vis;    
public:
    void dfs(vector<int>& nums,vector<int>& tmp,int begin,
    int target,int n,vector<vector<int>>& res){
        if(target<0) return;
        if(target==0) {
            res.push_back(tmp);
        }
        for(int i=begin;i<n;i++){
            if(i>begin&&nums[i]==nums[i-1]) continue;
            tmp.push_back(nums[i]);
            dfs(nums,tmp,i+1,target-nums[i],n,res);
            tmp.pop_back();
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> tmp;
        int n=candidates.size();
        vis.resize(candidates.size());
        sort(candidates.begin(),candidates.end());
        dfs(candidates,tmp,0,target,n,res);
        return res;
    }
};
```



# 贪心算法

## 45.跳跃游戏 II

[(https://leetcode.cn/problems/jump-game-ii/)

给定一个长度为 `n` 的 **0 索引**整数数组 `nums`。初始位置为 `nums[0]`。

每个元素 `nums[i]` 表示从索引 `i` 向前跳转的最大长度。

换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处:

```cpp
class Solution {  //贪心算法
public:
    //正向查找可以达到的最大位置
    //维护当前能够到达的最大下标位置，记为边界。
    //从左到右边遍历数组、到达边界时，更新边界并将跳跃次数加1
    int jump(vector<int>& nums) {
        int n=nums.size();
        int end=0,maxp=0,steps=0;
        for(int i=0;i<n-1;i++){   //
            if(maxp>=i){
                //当前跳数的最远跳跃点
                maxp=max(maxp,i+nums[i]);
                if(i==end){
                    //当前跳数的终点
                    end=maxp;
                    steps++;
                }
            }
        }
        return steps;
    }
};
```



## 55.跳跃游戏

[(https://leetcode.cn/problems/jump-game/)

给你一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

```cpp
class Solution {
public:
   //维护最大可达距离
    bool canJump(vector<int>& nums) {
        int n=nums.size();
        int rightmost=0;
        for(int i=0;i<n;i++){
            if(i<=rightmost){
                rightmost=max(rightmost,i+nums[i]);
                if(rightmost>=n-1){
                    return true;
                }
            }
        }
        return false;
    }
};
```



## 68.文本左右对齐

[(https://leetcode.cn/problems/text-justification/)

给定一个单词数组 `words` 和一个长度 `maxWidth` ，重新排版单词，使其成为每行恰好有 `maxWidth` 个字符，且左右两端对齐的文本。

你应该使用 “**贪心算法**” 来放置给定的单词；也就是说，尽可能多地往每行中放置单词。

必要时可用空格 `' '` 填充，使得每行恰好有 *maxWidth* 个字符。

要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。

文本的最后一行应为左对齐，且单词之间不插入**额外的**空格。

**注意:**

- 单词是指由非空格字符组成的字符序列。
- 每个单词的长度大于 0，小于等于 *maxWidth*。
- 输入单词数组 `words` 至少包含一个单词。

**示例 1:**

```
输入: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
输出:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
```

**示例 2:**

```
输入:words = ["What","must","be","acknowledgment","shall","be"], maxWidth = 16
输出:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
解释: 注意最后一行的格式应为 "shall be    " 而不是 "shall     be",
     因为最后一行应为左对齐，而不是左右两端对齐。       
     第二行同样为左对齐，这是因为这行只包含一个单词。
```

```cpp
class Solution {
    // blank 返回长度为 n 的由空格组成的字符串
    string blank(int n) {
        return string(n, ' ');
    }

    // join 返回用 sep 拼接 [left, right) 范围内的 words 组成的字符串
    string join(vector<string> &words, int left, int right, string sep) {
        string s = words[left];
        for (int i = left + 1; i < right; ++i) {
            s += sep + words[i];
        }
        return s;
    }

public:
    vector<string> fullJustify(vector<string> &words, int maxWidth) {
        vector<string> ans;
        int right = 0, n = words.size();
        while (true) {
            int left = right; // 当前行的第一个单词在 words 的位置
            int sumLen = 0; // 统计这一行单词长度之和
            // 循环确定当前行可以放多少单词，注意单词之间应至少有一个空格
            while (right < n && sumLen + words[right].length() + right - left <= maxWidth) {
                sumLen += words[right++].length();
            }

            // 当前行是最后一行：单词左对齐，且单词之间应只有一个空格，在行末填充剩余空格
            if (right == n) {
                string s = join(words, left, n, " ");
                ans.emplace_back(s + blank(maxWidth - s.length()));
                return ans;
            }

            int numWords = right - left;
            int numSpaces = maxWidth - sumLen;

            // 当前行只有一个单词：该单词左对齐，在行末填充剩余空格
            if (numWords == 1) {
                ans.emplace_back(words[left] + blank(numSpaces));
                continue;
            }

            // 当前行不只一个单词
            int avgSpaces = numSpaces / (numWords - 1);
            int extraSpaces = numSpaces % (numWords - 1);
            string s1 = join(words, left, left + extraSpaces + 1, blank(avgSpaces + 1)); // 拼接额外加一个空格的单词
            string s2 = join(words, left + extraSpaces + 1, right, blank(avgSpaces)); // 拼接其余单词
            ans.emplace_back(s1 + blank(avgSpaces) + s2);
        }
    }
};
```



# 动态规划

## 53.最大子数组和

[(https://leetcode.cn/problems/maximum-subarray/)

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组**

是数组中的一个连续部分。

*f*(*i*)=max{*f*(*i*−1)+*nums*[*i*],*nums*[*i*]}，当前子序列是选还是不选。

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len=nums.size();
        //dp[i]代表以 nums[i]结尾的连续子数组的最大和
        vector<int> dp;
        dp.push_back(nums[0]);
        int res=dp[0];
        for(int i=1;i<len;i++){
            if(dp[i-1]>0){
                dp.push_back(dp[i-1]+nums[i]);
            }else{
                dp.push_back(nums[i]);
            }
            res=max(res,dp[i]);
        }
        return res;
    }
};
```



## 63.不同路径 II (计数、滚动数组)

[(https://leetcode.cn/problems/unique-paths-ii/)

给定一个 `m x n` 的整数数组 `grid`。一个机器人初始位于 **左上角**（即 `grid[0][0]`）。

机器人尝试移动到 **右下角**（即 `grid[m - 1][n - 1]`）。机器人每次只能向下或者向右移动一步。

网格中的障碍物和空位置分别用 `1` 和 `0` 来表示。机器人的移动路径中不能包含 **任何** 有障碍物的方格。

返回机器人能够到达右下角的不同路径数量。

测试用例保证答案小于等于 `2 * 10*9`。

*f*(*i*,*j*) = 0                              *u*(*i*,*j*)=0

*f*(*i*,*j*)= f*(*i*−1,*j*)+*f*(*i*,*j−1)         *u*(*i*,*j*)!=0

```cpp
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int n = obstacleGrid.size(), m = obstacleGrid.at(0).size();
        vector <int> f(m);

        f[0] = (obstacleGrid[0][0] == 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (obstacleGrid[i][j] == 1) {
                    f[j] = 0;
                    continue;
                }
                if (j - 1 >= 0 && obstacleGrid[i][j - 1] == 0) {
                    f[j] += f[j - 1];
                }
            }
        }
        return f.back();
    }
};
```



## 64.最小路径和

[(https://leetcode.cn/problems/minimum-path-sum/)

给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，

请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        if (grid.size() == 0 || grid[0].size() == 0) {
            return 0;
        }
        int rows = grid.size(), columns = grid[0].size();
        auto dp = vector < vector <int> > (rows, vector <int> (columns));
        dp[0][0] = grid[0][0];
        for (int i = 1; i < rows; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < columns; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < columns; j++) {
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[rows - 1][columns - 1];
    }
};
```



## 87.扰乱字符串

[(https://leetcode.cn/problems/scramble-string/)

使用下面描述的算法可以扰乱字符串 `s` 得到字符串 `t` ：

1. 如果字符串的长度为 1 ，算法停止
2. 如果字符串的长度 > 1 ，执行下述步骤：
   - 在一个随机下标处将字符串分割成两个非空的子字符串。即，如果已知字符串 `s` ，则可以将其分成两个子字符串 `x` 和 `y` ，且满足 `s = x + y` 。
   - **随机** 决定是要「交换两个子字符串」还是要「保持这两个子字符串的顺序不变」。即，在执行这一步骤之后，`s` 可能是 `s = x + y` 或者 `s = y + x` 。
   - 在 `x` 和 `y` 这两个子字符串上继续从步骤 1 开始递归执行此算法。

给你两个 **长度相等** 的字符串 `s1` 和 `s2`，判断 `s2` 是否是 `s1` 的扰乱字符串。如果是，返回 `true` ；否则，返回 `false` 。

```cpp
class Solution {
private:
    //记忆化搜索存储状态的数组
    //-1表示false,1表示true,0表示未计算
    int memo[30][30][31];
    string s1,s2;
public:
    bool checkIfSimilar(int i1,int i2,int length){
        unordered_map<int,int> freq;
        for(int i=i1;i<i1+length;++i){
            ++freq[s1[i]];
        }
        for(int i=i2;i<i2+length;++i){
            --freq[s2[i]];
        }
        if(any_of(freq.begin(),freq.end(),[](const auto& entry){return entry.second!=0;})){
            return false;
        }
        return true;
    }

    //第一个字符串从i1开始，第二个字符串从i2开始，子串长度为length,是否和谐
    bool dfs(int i1,int i2,int length){
        if(memo[i1][i2][length]){
            return memo[i1][i2][length]==1;
        }
        //判断两个子串是否相等
        if(s1.substr(i1,length)==s2.substr(i2,length)){
            memo[i1][i2][length]=1;
            return true;
        }
        //判断是否存在字符c 在两个子串中出现的次数不同
        if(!checkIfSimilar(i1,i2,length)){
            memo[i1][i2][length]=-1;
            return false;
        }
        //枚举分割位置
        for(int i=1;i<length;++i){
            //不交换的情况
            if(dfs(i1,i2,i)&&dfs(i1+i,i2+i,length-i)){
                memo[i1][i2][length]=1;
                return true;
            }
            //交换的情况
            if(dfs(i1,i2+length-i,i)&&dfs(i1+i,i2,length-i)){
                memo[i1][i2][length]=1;
                return true;
            }
        }
        memo[i1][i2][length]=-1;
        return false;
    }

    bool isScramble(string s1, string s2) {
        memset(memo,0,sizeof(memo));
        this->s1=s1;
        this->s2=s2;
        return dfs(0,0,s1.size());
    }
};
```

## 91.解码方法



一条包含字母 `A-Z` 的消息通过以下映射进行了 **编码** ：

```
"1" -> 'A' "2" -> 'B' ... "25" -> 'Y' "26" -> 'Z'
```

然而，在 **解码** 已编码的消息时，你意识到有许多不同的方式来解码，因为有些编码被包含在其它编码当中（`"2"` 和 `"5"` 与 `"25"`）。

例如，`"11106"` 可以映射为：

- `"AAJF"` ，将消息分组为 `(1, 1, 10, 6)`
- `"KJF"` ，将消息分组为 `(11, 10, 6)`
- 消息不能分组为 `(1, 11, 06)` ，因为 `"06"` 不是一个合法编码（只有 "6" 是合法的）。

注意，可能存在无法解码的字符串。

给你一个只含数字的 **非空** 字符串 `s` ，请计算并返回 **解码** 方法的 **总数** 。

如果没有合法的方式解码整个字符串，返回 `0`。

题目数据保证答案肯定是一个 **32 位** 的整数。



```cpp
class Solution {
public:
    //设dp[i] 表示字符串的前i个字符s[1..i]的解码方法数
    //我们考虑两种情况：
    //A-使用一个字符即s[i]进行解码，只要s[i]!=0 则dp[i]=dp[i-1];
    //B-使用两个字符即s[i-1]和s[i]进行解码 要求s[i]和s[i-1]组成的整数必须小于等于26 大于0
    //则 dp[i]=dp[i-2];
    //对这两种情况进行累加
    int numDecodings(string s) {
        int n=s.size();
        vector<int> dp(n+1);
        dp[0]=1;//空字符串
        for(int i=1;i<=n;++i){
            if(s[i-1]!='0'){    //只取一个字符
                dp[i]+=dp[i-1];
            }
            if(i>1&&s[i-2]!='0'&&((s[i-2]-'0')*10+(s[i-1]-'0')<=26)){
                dp[i]+=dp[i-2];
            }
        }
        return dp[n];
    }
};
```



## 97.交错字符串

给定三个字符串 `s1`、`s2`、`s3`，请你帮忙验证 `s3` 是否是由 `s1` 和 `s2` **交错** 组成的。

两个字符串 `s` 和 `t` **交错** 的定义与过程如下，其中每个字符串都会被分割成若干 **非空**

![img](https://assets.leetcode.com/uploads/2020/09/02/interleave.jpg)

```cpp
class Solution {
public:
    //dp[i][j]表示s1的前i个元素和s2的前j个元素是否能够交错组成s3的前i+j个元素
    //如果s1的第i个元素和s3的第i+j个元素相等:
    //那么s1的前i个元素和s2的前j个元素是否能够交错组成s3的前i+j个元素
    //取决于s1的前i-1个元素和s2的前j个元素是否能交错组成s3的前i+j-1个元素
    //dp[i][j]=(dp[i-1][j]&&s1[i-1]==s3[p])||(dp[i][j-1]&&s2[j-1]==s3[p])
    //p=i+j-1 边界条件为 dp[0][0]=true;
    bool isInterleave(string s1, string s2, string s3) {
        auto dp=vector<vector<int>>(s1.size()+1,vector<int>(s2.size()+1,false));
        int n=s1.size(),m=s2.size(),t=s3.size();
        if(n+m!=t) return false;
        dp[0][0]=true;
        for(int i=0;i<=n;++i){
            for(int j=0;j<=m;++j){
                int p=i+j-1;
                if(i>0){
                    dp[i][j]|=(dp[i-1][j]&&s1[i-1]==s3[p]);
                }
                if(j>0){
                    dp[i][j]|=(dp[i][j-1]&&s2[j-1]==s3[p]);
                }
            }
        }
        return dp[n][m];
    }
};
```



# 合并

## 56.合并区间

[(https://leetcode.cn/problems/merge-intervals/)

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。

请你合并所有重叠的区间，并返回 *一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间* 。

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

```cpp
class Solution {
public:
    //先用左端点升序 然后将第一个区间加入到merged数组中，并按照次序考虑之后的区间：
    //如果当前区间的左端点在merged中最后一个区间的右端点之后，
    //--------------则不重合(我们直接将这个区间加入到数组merged末尾)
    //否则，重合(我们需要用当前区间的右端点更新数组merged中最后一个区间的右端点，
    //将其置为两者的最大值)
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if(intervals.size()==0) return{};
        //sort默认排序左端点升序
        sort(intervals.begin(),intervals.end());
        vector<vector<int>> merged;
        for(int i=0;i<intervals.size();i++){
            int L=intervals[i][0],R=intervals[i][1];
            //当前区间的左端点在merged中最后一个区间的右端点之后
            if(!merged.size()||merged.back()[1]<L){ 
                merged.push_back({L,R});
            }else{
                merged.back()[1]=max(merged.back()[1],R);
            }
        }
        return merged;
    }
};
```



## 57.插入区间

[(https://leetcode.cn/problems/insert-interval/)

给你一个 **无重叠的** *，*按照区间起始端点排序的区间列表 `intervals`，

其中 `intervals[i] = [starti, endi]` 表示第 `i` 个区间的开始和结束，并且 `intervals` 按照 `starti` 升序排列。

同样给定一个区间 `newInterval = [start, end]` 表示另一个区间的开始和结束。

在 `intervals` 中插入区间 `newInterval`，使得 `intervals` 依然按照 `starti` 升序排列，

且区间之间不重叠（如果有必要的话，可以合并区间）。

返回插入之后的 `intervals`。

**注意** 你不需要原地修改 `intervals`。你可以创建一个新数组然后返回它。

交集[max(*l1,*l*2),min(*r*1,*r2)]

并集[min(*l*1,*l*2),max(*r*1,*r*2)]

![fig1](https://assets.leetcode-cn.com/solution-static/57/1.png)



```cpp
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
          int l=newInterval[0];
          int r=newInterval[1];
          bool placed=false;
          vector<vector<int>> ans;
          for(const auto& interval:intervals){
              if(interval[0]>r){ //在插入区间右侧 且无交集
                  if(!placed){
                      ans.push_back({l,r});
                      placed=true;
                  }
                  ans.push_back(interval);
              }else if(interval[1]<l){ //在插入区间左侧 无交集
                  ans.push_back(interval);
              }else{// 有交集
                  l=min(l,interval[0]);
                  r=max(r,interval[1]);
              }
             
          } 
          if(!placed){
                   ans.push_back({l,r});
             }
          return ans;
    }
};
```



## 88.合并两个有序数组

[(https://leetcode.cn/problems/merge-sorted-array/)

给你两个按 **非递减顺序** 排列的整数数组 `nums1` 和 `nums2`，另有两个整数 `m` 和 `n` ，分别表示 `nums1` 和 `nums2` 中的元素数目。

请你 **合并** `nums2` 到 `nums1` 中，使合并后的数组同样按 **非递减顺序** 排列。

**注意：**最终，合并后数组不应由函数返回，而是存储在数组 `nums1` 中。

为了应对这种情况，`nums1` 的初始长度为 `m + n`，其中前 `m` 个元素表示应合并的元素，后 `n` 个元素为 `0` ，应忽略。

`nums2` 的长度为 `n` 。



```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        for(int i=0;i!=n;++i){
            nums1[m+i]=nums2[i];
        }
        sort(nums1.begin(),nums1.end());
    }
};
```





# 排序

## 75.颜色分类（冒泡排序）

给定一个包含红色、白色和蓝色、共 `n` 个元素的数组 `nums` ，**[原地](https://baike.baidu.com/item/原地算法)** 对它们进行排序，

使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 `0`、 `1` 和 `2` 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。



```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int p0 = 0, p2 = n - 1;
        for (int i = 0; i <= p2; ++i) {
            while (i <= p2 && nums[i] == 2) {
                swap(nums[i], nums[p2]);
                --p2;
            }
            if (nums[i] == 0) {
                swap(nums[i], nums[p0]);
                ++p0;
            }
        }
    }
};
```



```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int p0 = 0, p1 = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 1) {
                swap(nums[i], nums[p1]);
                ++p1;
            } else if (nums[i] == 0) {
                swap(nums[i], nums[p0]);
                if (p0 < p1) {
                    swap(nums[i], nums[p1]);
                }
                ++p0;
                ++p1;
            }
        }
    }
};
```



# 链表

## 86.分隔链表（双指针）

[(https://leetcode.cn/problems/partition-list/)

给你一个链表的头节点 `head` 和一个特定值 `x` ，请你对链表进行分隔，使得所有 **小于** `x` 的节点都出现在 **大于或等于** `x` 的节点之前。

你应当 **保留** 两个分区中每个节点的初始相对位置。

![img](https://assets.leetcode.com/uploads/2021/01/04/partition.jpg)

```cpp
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* small = new ListNode(0);
        ListNode* smallHead = small;
        ListNode* large = new ListNode(0);
        ListNode* largeHead = large;
        while (head != nullptr) {
            if (head->val < x) {
                small->next = head;
                small = small->next;
            } else {
                large->next = head;
                large = large->next;
            }
            head = head->next;
        }
        large->next = nullptr;
        small->next = largeHead->next;
        return smallHead->next;
    }
};
```



## 92.反转链表 II

[(https://leetcode.cn/problems/reverse-linked-list-ii/)

给你单链表的头指针 `head` 和两个整数 `left` 和 `right` ，其中 `left <= right` 。

请你反转从位置 `left` 到位置 `right` 的链表节点，返回 **反转后的链表** 。

![img](https://assets.leetcode.com/uploads/2021/02/19/rev2ex2.jpg)

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
private:
    void reverseLinkedList(ListNode *head){
        ListNode *pre=nullptr,*cur=head;
        while(cur!=nullptr){
            ListNode *next=cur->next;
            cur->next=pre;
            pre=cur;
            cur=next;
        }
    }
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        //虚拟头节点
        ListNode *dummyNode=new ListNode(-1);
        dummyNode->next=head;

        ListNode *pre=dummyNode;
        //来到left节点的前一个节点
        for(int i=0;i<left-1;i++){
            pre=pre->next;
        }
        ListNode *rightNode=pre;
        for(int i=0;i<right-left+1;i++){
            rightNode=rightNode->next;
        }
        //截取链表
        ListNode *leftNode=pre->next;
        ListNode *curr=rightNode->next;
        //切断
        pre->next=nullptr;
        rightNode->next=nullptr;
        //反转区间
        reverseLinkedList(leftNode);
        //还原链表
        pre->next=rightNode;
        leftNode->next=curr;
        return dummyNode->next;
    }
};
```





# 位域、二进制、解码

## 89.格雷编码

[(https://leetcode.cn/problems/gray-code/)

**n 位格雷码序列** 是一个由 `2n` 个整数组成的序列，其中：

- 每个整数都在范围 `[0, 2n - 1]` 内（含 `0` 和 `2n - 1`）
- 第一个整数是 `0`
- 一个整数在序列中出现 **不超过一次**
- 每对 **相邻** 整数的二进制表示 **恰好一位不同** ，且
- **第一个** 和 **最后一个** 整数的二进制表示 **恰好一位不同**

给你一个整数 `n` ，返回任一有效的 **n 位格雷码序列** 。



```cpp
class Solution {
public:
    
    vector<int> grayCode(int n) {
        vector<int> ret;
        ret.push_back(0);
        int head=1;
        for(int i=1;i<=n;i++){
            for(int j=ret.size()-1;j>=0;j--){
                ret.push_back(head+ret.at(j));
            }
            head<<=1;
        }
        return ret;
    }
};

//n=0   n=1   n=2   n=3    
// 0    0      00   0 00
//      1      01   0 01
//             11   0 11
//             10   0 10
//                  1 10
//                  1 11
//                  1 01
//                  1 00

//设n阶格雷码集合为G(n),G(n+1)阶格雷码为：
//给G(n)阶格雷码每个二进制形式前面添加0，得到G‘(n)
//设G(n)集合倒序(镜像)为R(n),给R(n)每个元素二进制形式前面添加1，得到R'(n)
//G(n+1)=G'(n)与R'(n)拼接两个集合即可得到下一阶格雷码
```

## 93.复原 IP 地址

[(https://leetcode.cn/problems/restore-ip-addresses/)

**有效 IP 地址** 正好由四个整数（每个整数位于 `0` 到 `255` 之间组成，且不能含有前导 `0`），整数之间用 `'.'` 分隔。

- 例如：`"0.1.2.201"` 和` "192.168.1.1"` 是 **有效** IP 地址，

  但是 `"0.011.255.245"`、`"192.168.1.312"` 和 `"192.168@1.1"` 是 **无效** IP 地址。

给定一个只包含数字的字符串 `s` ，用以表示一个 IP 地址，返回所有可能的**有效 IP 地址**，这些地址可以通过在 `s` 中插入 `'.'` 来形成。

你 **不能** 重新排序或删除 `s` 中的任何数字。你可以按 **任何** 顺序返回答案。



```cpp
//回溯法
class Solution {
public:
    vector<string> ans;
    vector<string> res;
    void backtrace(string& s,int i,int n,string cur){
        if(cur.size()>2||res.size()>4) return;
        if(res.size()==4&&i<n) return;
        if(res.size()==4&&i==n){ 
            string tmp;
            for(auto r:res){
                tmp+=r+'.';  
            }
             tmp.pop_back();
             ans.push_back(tmp);
            return;
        } 
        //选择当前字符作为cur最后一个字符
        cur.push_back(s[i]); 
        string tmp=cur;
        res.push_back(cur);
        if((cur.size()>1&&cur[0]=='0')||(cur.size()==3&&cur>"255")) {
            res.pop_back();
            return;
        }else{
            cur.clear();
            backtrace(s,i+1,n,cur);
            res.pop_back();
            cur=tmp;
        }
        //不选择当前字符作为cur的最后一个字符
        backtrace(s,i+1,n,cur);
    }

    vector<string> restoreIpAddresses(string s) {
        string cur="";
        int n=s.size();
        backtrace(s,0,n,cur);
        return ans;
    }
};
```

# 集合

## 90.子集 II

[(https://leetcode.cn/problems/subsets-ii/)

给你一个整数数组 `nums` ，其中可能包含重复元素，请你返回该数组所有可能的 

子集（幂集）。

解集 **不能** 包含重复的子集。返回的解集中，子集可以按 **任意顺序** 排列。



### A递归法

```cpp
class Solution {
public:
    vector<int> t;
    vector<vector<int>> ans;

    void dfs(bool choosePre,int cur,vector<int>& nums){
        if(cur==nums.size()){
            ans.push_back(t);
            return;
        }
        dfs(false,cur+1,nums); //不选择当前数
        //[1,2,2,2,2,2]
        //[1,x,2]=[1,2,x]
        //[1,2] [1,2,2],[1,2,2,2]
        //因为前面的数已经选择的版本会与选择当前的数产生重复
        //规则：同序列中如果前一个不选，则后面都不选
        //     如果前一个选择，则后面一个可以考虑选择或者不选
        //     这样就可以去掉重复的子序列
        //前面的数没有选择，且前面的数与当前的数相同，则跳过本次选择
        if(!choosePre&&cur>0&&nums[cur-1]==nums[cur]) return;
        t.push_back(nums[cur]);
        dfs(true,cur+1,nums); //选择当前数
        t.pop_back();   
    }

    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        dfs(false,0,nums);
        return ans;
    }
};
```

### B位域法

```cpp
class Solution {
public:
    vector<int> t;
    vector<vector<int>> ans;

    //注意 [1,2,2] 选择第一、第三个数 与 选择第一、第二个数都可以得到相同子集
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        int n=nums.size();
        for(int mask=0;mask<(1<<n);++mask){
            t.clear();
            bool flag=true;
            //先生成位数集合
            //第n位的数表示第n个元素是否取入子集合
            for(int i=0;i<n;++i){
                //查看当前i是否与当前mask位数匹配
                if(mask&(1<<i)){
                    //若发现选择没有上一个数，且当前数字与上一个数相同
                    //则可以跳过当前生成的子集
                    if(i>0&&(mask>>(i-1)&1)==0&&nums[i]==nums[i-1]){
                        flag=false; break;
                    }
                    t.push_back(nums[i]);
                }
            }
            if(flag) ans.push_back(t);
        }
        return ans;
    }
};
```



# 二叉树

## 95.不同的二叉搜索树 II

[(https://leetcode.cn/problems/unique-binary-search-trees-ii/)

给你一个整数 `n` ，请你生成并返回所有由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的不同 **二叉搜索树** 。

可以按 **任意顺序** 返回答案。

![img](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*>generateTrees(int start,int end){
        if(start>end) return {nullptr};
        vector<TreeNode*> allTrees;
        for(int i=start;i<=end;i++){
            //所有可行的左子树集合
            vector<TreeNode*> leftTrees=generateTrees(start,i-1);
            //所有可行的右子树集合
            vector<TreeNode*> rightTrees=generateTrees(i+1,end);
            // 从左子树集合中选出一棵左子树，从右子树集合中选出一棵右子树，拼接到根节点上
            for(auto& left:leftTrees){
                for(auto& right:rightTrees){
                    TreeNode* currTree=new TreeNode(i);
                    currTree->left=left;
                    currTree->right=right;
                    allTrees.emplace_back(currTree);
                }
            }
        }
        return allTrees;
    }

    vector<TreeNode*> generateTrees(int n) {
        if(!n) return {};
        return generateTrees(1,n);
    }
};
```



## 96.不同的二叉搜索树

给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？

返回满足题意的二叉搜索树的种数。

![img](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)



```cpp
class Solution {
public:
    //G(n)： 长度为n的序列能构成的不同二叉搜索树的个数
    //F(i,n): 以i为根、序列长度为n的不同二叉搜索树个数(1<=i<=n)
    //F(i,n)对应的每个二叉搜索树具有唯一性
    //F(i,n)=G(i-1)*G(n-i)
    //G(n)=E(n...1)i=1[G(i-1)*G(n-i)]
    int numTrees(int n) {
        vector<int> G(n+1,0);
        G[0]=1;
        G[1]=1;

        for(int i=2;i<=n;i++){
            for(int j=1;j<=i;j++){
                G[i]+=G[j-1]*G[i-j];
            }
        }
        return G[n];
    }
};
```



## 98.验证二叉搜索树

[(https://leetcode.cn/problems/validate-binary-search-tree/)

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 小于 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */

//二叉搜索树中：
//左子树的所有结点都小于当前根
//右子树的所有结点都大于当前根
//后序遍历
//使用递归函数helper(TreeNode* root,long long lower,long long upper)
//其中 lower upper 对应开区间(l,r)
//返回判断当前root是否在(l,r)区间内


//          5
//      1       7
//           6     8    
//  7所在区间为(5,long max)
//  6所在区间为(5,7)

class Solution {
public:
    bool helper(TreeNode* root,long long lower,long long upper){
        if(root==nullptr)  return true;
        if(root->val<=lower||root->val>=upper) return false;
        return helper(root->left,lower,root->val)&&helper(root->right,root->val,upper);
    }
    bool isValidBST(TreeNode* root) {
        return helper(root,LONG_MIN,LONG_MAX);
    }
};

```



## 99.恢复二叉搜索树

[(https://leetcode.cn/problems/recover-binary-search-tree/)

给你二叉搜索树的根节点 `root` ，该树中的 **恰好** 两个节点的值被错误地交换。

*请在不改变其结构的情况下，恢复这棵树* 。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void inorder(TreeNode *root,vector<int>& nums){
        if(root==nullptr) return;
        inorder(root->left,nums);
        nums.push_back(root->val);
        inorder(root->right,nums);
    }

    pair<int,int> findTwoSwapped(vector<int>&  nums){
        int n=nums.size();
        int index1=-1,index2=-1;
        for(int i=0;i<n-1;++i){
            if(nums[i+1]<nums[i]){
                index2=i+1;
                if(index1==-1){
                    index1=i;
                }else{
                    break;
                }
            }
        }
        int x=nums[index1],y=nums[index2];
        return {x,y};
    }

    void recover(TreeNode *r,int count,int x,int y){
        if(r!=nullptr){
            if(r->val==x||r->val==y){
                r->val=r->val==x?y:x;
                if(--count==0) return;
            }
            recover(r->left,count,x,y);
            recover(r->right,count,x,y);
        }
    }

    void recoverTree(TreeNode* root) {
        vector<int> nums;
        inorder(root,nums);
        pair<int,int> swapped=findTwoSwapped(nums);
        recover(root,2,swapped.first,swapped.second);
    }
};
```

## 100.相同的树

[(https://leetcode.cn/problems/same-tree/)

给你两棵二叉树的根节点 `p` 和 `q` ，编写一个函数来检验这两棵树是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(!p&&!q) return true;
        if(((!p&&q)||(p&&!q)||(p->val!=q->val))) return false;
        return isSameTree(p->left,q->left)&&isSameTree(p->right, q->right);
    }
};
```
