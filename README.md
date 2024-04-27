leetcode--notes
===
Task one Data Structure and Algorithm
------
## Data Sructure :带有结构特性的数据元素的集合

>逻辑结构（Logical Structure）：数据元素之间的相互关系
根据元素之间具有的不同关系，通常我们可以将数据的逻辑结构分为以下四种：
* 1. 集合结构:数据元素同属于一个集合，除此之外无其他关系。
* 2. 线性结构:数据元素之间是「一对一」关系
* 3. 树形结构:数据元素之间是「一对多」的层次关系。
* 4. 图形结构:数据元素之间是[多对多]的关系

>物理结构（Physical Structure）：数据的逻辑结构在计算机中的存储方式
* 1.顺序存储结构（Sequential Storage Structure）：将数据元素存放在一片地址连续的存储单元里，数据元素之间的逻辑关系通过数据元素的存储地址来直接反映。
 * 2.链式存储结构（Linked Storage Structure）：将数据元素存放在任意的存储单元里，存储单元可以连续，也可以不连续。

## 算法（Algorithm）：解决特定问题求解步骤的准确而完整的描述，在计算机中表现为一系列指令的集合，算法代表着用系统的方法描述解决问题的策略机制。
算法的基本特性:
>* 1.输入：对于待解决的问题，都要以某种方式交给对应的算法。在算法开始之前最初赋给算法的参数称为输入。比如示例 
 中的输入就是出发地和目的地的参数（北京，上海），示例 
 中的输入就是 
 个整数构成的数组。一个算法可以有多个输入，也可以没有输入。比如示例 
 是对固定问题的求解，就可以看做没有输入。
>* 2.输出：算法是为了解决问题存在的，最终总需要返回一个结果。所以至少需要一个或多个参数作为算法的输出。比如示例 
 中的输出就是最终选择的交通方式，示例 
 中的输出就是和的结果。示例 
 中的输出就是排好序的数组。
>* 3.有穷性：算法必须在有限的步骤内结束，并且应该在一个可接受的时间内完成。<br>
>* 4.确定性：组成算法的每一条指令必须有着清晰明确的含义，不能令读者在理解时产生二义性或者多义性。就是说，算法的每一个步骤都必须准确定义而无歧义。<br>
>* 5.可行性：算法的每一步操作必须具有可执行性，在当前环境条件下可以通过有限次运算实现。也就是说，每一步都能通过执行有限次数完成，并且可以转换为程序在计算机上运行并得到正确的结果。<br>

算法追求的目标
>* 1.所需运行时间更少（时间复杂度更低）；
>* 2.占用内存空间更小（空间复杂度更低）。<br>

一个好的算法还应该追求以下目标：
>* 1.正确性：正确性是指算法能够满足具体问题的需求，程序运行正常，无语法错误，能够通过典型的软件测试，达到预期的需求。
>* 2.可读性：可读性指的是算法遵循标识符命名规则，简洁易懂，注释语句恰当，方便自己和他人阅读，便于后期修改和调试。
>* 3.健壮性：健壮性指的是算法对非法数据以及操作有较好的反应和处理。

常根据从小到大排序，常见的算法复杂度主要有：O（1）< O(log n) < O(n) < O(n²) < O(2的n次方)等。
常见的时间复杂度有：O(1),O(log n),O(n),O(n*log n),O(n²),O(n³),O(2的n次方),O(n!)
常见的空间复杂度有：O(1),O(log n),O(n),O(n²)

# LeetCode是什么
  `「LeetCode」` 是一个代码在线评测平台（Online Judge），包含了 算法、数据库、Shell、多线程 等不同分类的题目，其中以算法题目为主。我们可以通过解决 LeetCode 题库中的问题来练习编程技能，以及提高算法能力。
  LeetCode 上有多道的编程问题，支持多种种编程语言（C、C++、Java、Python 等），还有一个活跃的社区，可以用于分享技术话题、职业经历、题目交流等。
  并且许多知名互联网公司在面试的时候喜欢考察 LeetCode 题目，通常会以手写代码的形式出现。需要面试者对给定问题进行分析并给出解答，有时还会要求面试者分析算法的时间复杂度和空间复杂度，以及算法思路。面试官通过考察面  试者对常用算法的熟悉程度和实现能力来确定面试者解决问题的思维能力水平。
  所以无论是面试国内还是国外的知名互联网公司，通过 LeetCode 刷题，充分准备好算法，对拿到一个好公司的好 offer 都是有帮助的。
## LeetCode新手注册
1.打开 LeetCode 中文主页，链接:[LeetCode](https://leetcode.cn/)
常考的数据结构：数组、字符串、链表、树（如二叉树） 等。
常考的算法：分治算法、贪心算法、穷举算法、回溯算法、动态规划 等

# LeetCode day 1
1.[两整数相加](https://leetcode.cn/problems/add-two-integers/description/)
```
class Solution(object):
    def sum(self, num1, num2):
        return num1+num2
```
2.[数组串联](https://leetcode.cn/problems/concatenation-of-array/description/)
```
class Solution(object):
    def getConcatenation(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        ans=nums+nums
        return ans
```
3.[宝石与石头](https://leetcode.cn/problems/jewels-and-stones/description/)
```
class Solution(object):
    def numJewelsInStones(self, jewels, stones):
        """
        :type jewels: str
        :type stones: str
        :rtype: int
        """
        count=0
        for i in stones:
            for j in jewels:
                if i==j :
                    count=count+1
        return count
```
# LeetCode day 2
1.[ 一维数组的动态和](https://leetcode.cn/problems/running-sum-of-1d-array/description/)
```
class Solution(object):
    def runningSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        Sum=[]
        for i in range(len(nums)):
            Sum.append(sum(nums[:i+1]))
        return Sum
```
2.[转换成小写字母](https://leetcode.cn/problems/to-lower-case/description/)
```
class Solution(object):
    def toLowerCase(self, s):
        """
        :type s: str
        :rtype: str
        """
        s=s.lower()
        return s

```
3.[最富有客户的资产总量](https://leetcode.cn/problems/richest-customer-wealth/description/)
```
class Solution(object):
    def maximumWealth(self, accounts):
        """
        :type accounts: List[List[int]]
        :rtype: int
        """
        rich=[]
        for i in range(len(accounts)):
                rich.append(sum(accounts[i]))
        return max(rich)
```


# Task 2
数组基础：
数组：线性表结构。用一系列连续的内存空间来存储具有同一类型的数据
索引：数组中的每一个元素都具有自己的下标索引，下标索引从0开始到n-1结束
数组的特点：1.线性表：所有元素排成一条线一样的结构且每个元素只有前跟后两个方向
            2.连续的存储空间：线性表有两种存储结构分别是顺序存储结构和链式存储结构，顺序存储结构指占用的内存空间是连续的，相邻数据元素之间，在内存上的物理位置也相邻。数组就是采用的顺序存储结构
如何访问数组中的元素
通过下标索引，可以直接定位到元素存放的位置
寻址公式如下：下标i对应的数据元素地址 =数据首地址 +i × 单个数据元素所占内存大小。
二维数组：由m行n列数组元素构成的特殊结构，本质是以数组作为元素的数组，称之为数组中的数组，二维数组可以用来处理矩阵等相关问题，比如转置矩阵，矩阵相加，矩阵相乘等。
数组的基本操作：增，删，改，查
## 练习题目
1.[0066.加一](https://leetcode.cn/problems/plus-one/description/)
```
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        val = 0
        if digits[len(digits)-1]==9:
            digits[len(digits)-1]=1
            digits.append(val)
        else: digits[len(digits)-1] += 1 
        return digits
```
2.[0724.寻找数组中心下标](https://leetcode.cn/problems/find-pivot-index/description/)
```
class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        N = len(nums)
        sums_ = sum(nums)
        preSum = 0
        for i in range(N):
            if preSum ==sums_ - preSum -nums[i]:
                return i
            preSum += nums[i]
        return -1
```
3.[0189.轮转数组](https://leetcode.cn/problems/rotate-array/description/)
```
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if k := (k % len(nums)):
            nums[:k], nums[k:] = nums[-k:], nums[:-k]
```
## 练习题目第4天
1.[0048.旋转图像](https://leetcode.cn/problems/rotate-image/description/)
```
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n//2):
            for j in range((n+1)//2):
                tmp = matrix[i][j]
                matrix[i][j]=matrix[n-1-j][i]
                matrix[n-1-j][i]=matrix[n-1-i][n-1-j]
                matrix[n-1-i][n-1-j]=matrix[j][n-1-i]
                matrix[j][n-1-i]=tmp
```
2.[0054.螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/description/)
```
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix: return []
        l,r,t,b,res = 0,len(matrix[0])-1,0,len(matrix)-1,[]
        while True:
            for i in range(l,r+1): res.append(matrix[t][i])
            t += 1
            if t>b: break
            for i in range(t,b+1): res.append(matrix[i][r])
            r-=1
            if l>r: break
            for i in range(r,l-1,-1): res.append(matrix[b][i])
            b -= 1
            if t>b: break
            for i in range(b,t-1,-1):res.append(matrix[i][l])
            l += 1
            if l>r: break
        return res
```
3.[0498.对角线遍历](https://leetcode.cn/problems/diagonal-traverse/description/)
```
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        ans = []
        m,n = len(mat), len(mat[0])
        for i in range(m+n-1):
            if i%2:
                x = 0 if i < n else i-n+1
                y = i if i < n else n-1
                while x <m and y>=0:
                    ans.append(mat[x][y])
                    x += 1
                    y -= 1
            else:
                x = i if i < m else m - 1
                y = 0 if i < m else i-m+1
                while x>=0 and y < n:
                    ans.append(mat[x][y])
                    x-= 1
                    y+= 1
        return ans
```
## 练习题目第四天
1.[LCR 164. 破解闯关密码](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/description/)
```
class Solution:
    def crackPassword(self, password: List[int]) -> str:
        def quick_sort(l,r):
            if l>=r: return
            i,j = l, r
            while i < j:
                while strs[j] + strs[l] >= strs[l] + strs[j] and i < j: j -=1
                while strs[i] + strs[l] <= strs[l] + strs[j] and i < j: i +=1
                strs[i], strs[j] = strs[j], strs[i]
            strs[i], strs[l] = strs[l], strs[i]
            quick_sort(l,i-1)
            quick_sort(i+1,r)
        
        strs = [str(num) for num in password]
        quick_sort(0, len(strs)-1)
        return ''.join(strs)
```
2.[283.移动零](https://leetcode.cn/problems/move-zeroes/description/)
```
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = 0
        for right in range(len(nums)):
            if nums[right]:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
```
3.[912.排序数组](https://leetcode.cn/problems/sort-an-array/description/)
```
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return self.mergeSort(nums)
    def merge(self, left_nums: [int], right_nums: [int]):
                nums = []
                left_i, right_i = 0, 0
                while left_i< len(left_nums) and right_i< len(right_nums):
                    if left_nums[left_i] < right_nums[right_i]:
                        nums.append(left_nums[left_i])
                        left_i += 1
                    else:
                        nums.append(right_nums[right_i])
                        right_i += 1
                while left_i < len(left_nums):
                    nums.append(left_nums[left_i])
                    left_i += 1
                while right_i < len(right_nums):
                    nums.append(right_nums[right_i])
                    right_i += 1
                return nums
    def mergeSort(self, nums: [int]) -> [int]:
            if len(nums) <= 1:
                return nums

            mid = len(nums) // 2
            left_nums = self.sortArray(nums[:mid])
            right_nums = self.sortArray(nums[mid:])
            return self.merge(left_nums, right_nums)
```
## 练习题目第06天
1.[506.相对名次](https://leetcode.cn/problems/relative-ranks/description/)
```
class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        n = len(score)
        ans = list()
        res = score.copy()
        res = Solution().sortArray(res)
        for item in score:
            j = score.index(item)
            k = res.index(item)
            if k == 0:
                ans.insert(j, "Gold Medal")
            elif k == 1:
                ans.insert(j, "Silver Medal")
            elif k == 2:
                ans.insert(j, "Bronze Medal")
            else:
                ans.insert(j,f"{k + 1}")
        return ans
    def randomPartition(self, nums: [int], low: int, high:int) -> int:
        i = random(low, high)
        nums[i], nums[0] = nums[0], nums[i]
        return self.Partition(nums, low, high)
    def Partition(self, nums: [int], low:int, high: int) -> int:
        pivot = nums[low]

        i,j = low,high
        while i<j:
                while i < j and nums[j] <= pivot:
                    j -= 1
                while i < j and nums[i] >= pivot:
                    i += 1
                nums[i], nums[j] = nums[j], nums[i]
        nums[j], nums[low] = nums[low], nums[j]
        return j
    def quickSort(self, nums: [int], low: int, high: int) -> int:
        if low < high:
            pivot_i = self.Partition(nums,low,high)
            self.quickSort(nums,low,pivot_i - 1)
            self.quickSort(nums,pivot_i + 1, high)
        
        return nums
    def sortArray(self, nums: [int]) -> [int]:
        return self.quickSort(nums, 0, len(nums) - 1)
```
$\sum_{i=1}^{n}{(i-1)}$ = $\frac {n(n+1)}{2}$

$\int_1^\infty$
