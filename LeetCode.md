---
layout: post
title: LeetCode
---

## 1. Two Sum
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.

> Given nums = [2, 7, 11, 15], target = 9. Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        index_dict = {}
        for i in range(len(nums)):
            if nums[i] in index_dict.keys():
                return [i,index_dict[nums[i]]]
            else:
                index_dict[target-nums[i]] = i
```

## 2. Add Two Numbers

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

> Input: (2 -> 4 -> 3) + (5 -> 6 -> 4), Output: 7 -> 0 -> 8


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        carryOver = 0
        sum_list = []

        while l1 != None or l2 != None:
            if l1 == None:
                add = l2.val + carryOver
                digit = add % 10
                carryOver = add / 10
                l2 = l2.next
            elif l2 == None:
                add = l1.val + carryOver
                digit = add % 10
                carryOver = add / 10
                l1 = l1.next
            else:
                add = l1.val + l2.val + carryOver
                digit = add % 10
                carryOver = add / 10
                l1 = l1.next
                l2 = l2.next

            sum_list.append(digit)
        if carryOver != 0:
            sum_list.append(carryOver)

        return sum_list
```

## 3. Longest Substring Without Repeating Characters
Given a string, find the length of the longest substring without repeating characters.

> Examples:

> Given "abcabcbb", the answer is "abc", which the length is 3.

> Given "bbbbb", the answer is "b", with the length of 1.

> Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.


```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        dictionary = {}
        L = len(s)
        start = 0
        LSWRC = 0 # Longest substring without repeating characters

        for idx in range(L):
            if s[idx] not in dictionary:
                dictionary[s[idx]] = idx
            else:
                if dictionary[s[idx]] < start:
                    dictionary[s[idx]] = idx
                else:
                    LSWRC = max(LSWRC,idx-start)
                    start = dictionary[s[idx]] + 1
                    dictionary[s[idx]] = idx
        if L > 0:        
            LSWRC = max(LSWRC,idx+1-start)

    return LSWRC
```

## 4. Median of Two Sorted Arrays
There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

> Example 1:
nums1 = [1, 3]
nums2 = [2]
The median is 2.0

> Example 2:
nums1 = [1, 2]
nums2 = [3, 4]
The median is (2 + 3)/2 = 2.5


```python
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """

        median_index = (len(nums1) + len(nums2))/2
        import math
        median_indices = [math.floor(median_index), math.ceil(median_index)]

        if len(nums1) == 0:
            return mean([nums2[i] for i in median_indices])
        if len(nums2) == 0:
            return mean([nums1[i] for i in median_indices])

        idx = median_index
        print idx
        while idx > 0:
            index1 = int(min(len(nums1),median_index/2))
            index2 = int(min(len(nums2),median_index/2 - index1))
            index1 = int(min(len(nums1),median_index/2 - index2))

            print index1, index2, nums1, nums2, idx
            if len(nums1) == 0:
                return nums2[index2]
            elif len(nums2) == 0:
                return nums1[index1]
            elif nums1[index1] < nums2[index2]:
                nums1 = nums1[index1+1:]
                nums2 = nums2[:index2+1]
                idx -= index1
            elif nums1[index1] > nums2[index2]:
                nums1 = nums1[:index1+1]
                nums2 = nums2[index2+1:]
                idx -= index2
            else:
                print nums1, nums2

            median_index /= 2
```

## 5. Longest Palindromic Substring
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

>Example 1:
```
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
```

>Example 2:
```
Input: "cbbd"
Output: "bb"
```

```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        L = len(s)
        if L == 0:
            return ''

        isPalindrome = [[False for _ in range(L)] for _ in range(L)]
        solution = s[0]
        
        for i in range(L):
            isPalindrome[i][i] = True
        
        diff = 1
        for i in range(L-1):
            if s[i] == s[i+diff]:
                isPalindrome[i][i+diff] = True
                solution = s[i:i+diff+1]
                
        for diff in range(2,L):
            for i in range(L-diff):
                j = i + diff
                if s[i] == s[j] and isPalindrome[i+1][j-1] == True:
                    solution = s[i:j+1]
                    isPalindrome[i][j] = True
                else:
                    isPalindrome[i][j] = False
        return solution
```

## 6. ZigZag Conversion
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
```
P   A   H   N
A P L S I I G
Y   I   R
```
And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:
```
string convert(string s, int numRows);
```
>Example 1:
```
Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
```
>Example 2:
```
Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I
```

```python
class Solution:
    # @param {string} s
    # @param {integer} numRows
    # @return {string}
    def convert(self, s, numRows):
        L = len(s)
        s_zigzag = ''
        if L == 0:
            return ''
        if numRows == 1:
            return s

        # pivots indicate the index of the top row elements. We use at least one extra pivot to accomodate all cases.
        pivots = range(0,L+2*numRows-2,2*numRows-2)
        for p in pivots:
            s_zigzag +=  self.get_char(s,L,p)
        for diff in range(1,numRows-1):
            s_zigzag += self.get_char(s,L,pivots[0]+diff)
            for p in pivots[1:]:
                s_zigzag += self.get_char(s,L,p-diff)
                s_zigzag += self.get_char(s,L,p+diff)
        for p in pivots:
            s_zigzag += self.get_char(s,L,p+numRows-1)
        return s_zigzag
    
    def get_char(self,s,L,index):
        if index > L-1:
            return ''
        return s[index]
```

## 7. Reverse Integer
Reverse digits of an integer.

>Example1: x = 123, return 321

>Example2: x = -123, return -321


```python
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        s = str(x)
        if s[0] == '-':
            s_reversed = '-'+s[:0:-1]
        else:    
            s_reversed = s[::-1]

        solution = int(s_reversed)
        if solution > 2**31 or solution < -2**31:
            solution = 0

        return solution
```

## 9. Palindrome Number
Determine whether an integer is a palindrome. Do this without extra space.

```python
class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        return str(x) == str(x)[::-1]
```

## 11. Container With Most Water
Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # Use invariants
        i = 0
        j = len(height)-1
        solution = 0
        while i < j:
            solution = max(solution, (j-i)*min(height[i],height[j]))
            if height[i] < height[j]:
                i += 1
            elif height[i] > height[j]:
                j -= 1
            elif height[i] == height[j]:
                i += 1
                j -= 1

        return solution
```

## 12. Integer to Roman
Given an integer, convert it to a roman numeral.

Input is guaranteed to be within the range from 1 to 3999.

```python
class Solution:
    # @param {integer} num
    # @return {string}
    def intToRoman(self, num):
        thousands = num / 1000
        hundreds = (num - thousands*1000) / 100
        tens = (num - thousands*1000 - hundreds*100) / 10
        units = num - thousands*1000 - hundreds*100 - tens*10

        output = ''
        output += 'M'*thousands

        if hundreds <= 3 or (hundreds > 5 and hundreds <= 8):
            output += 'D'*(hundreds>5)+'C'*(hundreds % 5)
        elif hundreds == 4:
            output += 'CD'
        elif hundreds == 5:
            output += 'D'
        else: #hundreds == 9:
            output += 'CM'

        if tens <= 3 or (tens > 5 and tens <= 8):
            output += 'L'*(tens>5)+'X'*(tens % 5)
        elif tens == 4:
            output += 'XL'
        elif tens == 5:
            output += 'L'
        else: #tens == 9:
            output += 'XC'

        if units <= 3 or (units > 5 and units <= 8):
            output += 'V'*(units>5)+'I'*(units % 5)
        elif units == 4:
            output += 'IV'
        elif units == 5:
            output += 'V'
        else: #units == 9:
            output += 'IX'            

        return output
```

## 13. Roman to Integer
Given a roman numeral, convert it to an integer.

Input is guaranteed to be within the range from 1 to 3999.


```python
class Solution:
    # @param {string} s
    # @return {integer}
    def romanToInt(self, s):
        output = 0
        for i in range(0,len(s)):
            if s[i] == 'M':
                output += 1000
            elif s[i] == 'D':
                output += 500
            elif s[i] == 'L':
                output += 50
            elif s[i] == 'V':
                output += 5
            elif s[i] == 'C':
                if i != len(s)-1 and (s[i+1] == 'M' or s[i+1] == 'D'):
                    output -= 100
                else:
                    output += 100
            elif s[i] == 'X':
                if i != len(s)-1 and (s[i+1] == 'C' or s[i+1] == 'L'):
                    output -= 10
                else:
                    output += 10
            elif s[i] == 'I':
                if i != len(s)-1 and (s[i+1] == 'X' or s[i+1] == 'V'):
                    output -= 1
                else:
                    output += 1                    

        return output
```

## 14. Longest Common Prefix
Write a function to find the longest common prefix string amongst an array of strings.


```python
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        condition = True
        index = 0
        solution = ''
        if strs == []:
            return solution
        elif len(strs) == 1:
            return strs[0]

        while condition:
            if index >= len(strs[0]):
                condition = False
                break
            char = strs[0][index]
            for str in strs[1:]:
                if index >= len(str):
                    condition = False
                    return solution
                if str[index] != char:
                    condition = False
                    return solution
            solution += char
            index += 1
        return solution
```

## 15. 3Sum
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note: The solution set must not contain duplicate triplets.

>For example, given array S = [-1, 0, 1, 2, -1, -4], A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]


```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """        
        # Solve using two-sum
        nums = sorted(nums)
        
        L = len(nums)
        solution = []
        tried = {}
        
        for i in range(L):
            if nums[i] not in tried:
                for s in self.twoSum(nums[i+1:], -nums[i]):
                    solution += [[nums[i]]+s]
            tried[nums[i]] = 0
            
        return solution
        
    def twoSum(self,nums,target):
        solution = []
        dictionary = {}
        L = len(nums)
        
        for i in range(L):
            if target-nums[i] in dictionary:
                minimum = min(target-nums[i], nums[i])
                maximum = max(target-nums[i], nums[i])
                if [minimum,maximum] not in solution:
                    solution += [[minimum, maximum]]
            dictionary[nums[i]] = 0
            
        return solution
            

# Second method - first sort array and use invariants
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums = sorted(nums)
        n = len(nums)
        solutions = {}
        for i in range(n-2):
            a = nums[i]
            first = i+1
            last = n-1
            while first < last:
                b = nums[first]
                c = nums[last]
                if a+b+c == 0:
                    if (a,b,c) not in solutions: # avoid duplicates
                        solutions[(a,b,c)] = 0
                    # Continue search for more combinations
                    last -= 1
                elif a+b+c > 0:
                    last -= 1
                else:
                    first += 1
        return solutions.keys()
```

## 16. 3Sum Closest
Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

> For example, given array S = {-1 2 1 -4}, and target = 1. The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).


```python
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums = sorted(nums)
        L = len(nums)
        solution = nums[0] + nums[1] + nums[2] # initialization

        for i in range(L-2):
            a = nums[i]
            first = i+1
            last = L-1
            while first < last:
                b = nums[first]
                c = nums[last]
                if a+b+c-target == 0:
                    return target
                elif a+b+c-target > 0:
                    if abs(solution-target) > abs(a+b+c-target):
                        solution = a+b+c
                    last -= 1
                else:
                    if abs(solution-target) > abs(a+b+c-target):
                        solution = a+b+c                 
                    first += 1

        return solution
```

## 19. Remove Nth Node From End of List
Given a linked list, remove the nth node from the end of list and return its head.

> For example, Given linked list: 1->2->3->4->5, and n = 2. After removing the second node from the end, the linked list becomes 1->2->3->5.

Note:
Given n will always be valid.
Try to do this in one pass.


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        index = 0
        tailNode = head
        headNode = head

        if head.next == None:
            return []

        while tailNode.next != None:
            tailNode = tailNode.next
            if index == n:
                headNode = head
            if index >= n:
                headNode = headNode.next
            index += 1

        if n == 1:
            headNode.next = None
        elif index == n-1:
            return head.next
        else:                
            headNode.next = headNode.next.next

        return head

```

## 20. Valid Parentheses
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.



```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # Idea: use a stack. We can simply use a list in python as a proxy
        if len(s) == 0:
            return True

        symbols_dict = {}    
        symbols_dict['('] = 1
        symbols_dict['{'] = 2
        symbols_dict['['] = 3
        symbols_dict[')'] = -1
        symbols_dict['}'] = -2
        symbols_dict[']'] = -3

        if symbols_dict[s[0]] < 0:
            return False

        symbols_stack = [s[0]]
        for i in range(1,len(s)):
            if symbols_stack == []:
                if symbols_dict[s[i]] < 0:
                    return False
                symbols_stack.append(s[i])
            else:
                if symbols_dict[s[i]] == -symbols_dict[symbols_stack[-1]]: # matching parantheses
                    symbols_stack = symbols_stack[:-1]
                elif symbols_dict[s[i]] * symbols_dict[symbols_stack[-1]] > 0: # opening or closing parantheses
                    symbols_stack.append(s[i])        
                else:
                    return False
        if symbols_stack == []:
            return True
        else:
            return False

```

## 21. Merge Two Sorted Lists
Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 == None:
            return l2
        elif l2 == None:
            return l1

        l3 = ListNode(0)
        initialized = 0

        while l1 != None or l2 != None:
            if l1 == None:
                next_val = l2.val
                l2 = l2.next
            elif l2 == None:
                next_val = l1.val
                l1 = l1.next
            elif l2.val < l1.val:
                next_val = l2.val
                l2 = l2.next                
            elif l1.val <= l2.val:
                next_val = l1.val
                l1 = l1.next                

            if not initialized:
                l3 = ListNode(next_val)
                initialized = 1
                l3_copy = l3
            else:
                l3_copy.next = ListNode(next_val)
                l3_copy = l3_copy.next

        return l3
```

## 22. Generate Parentheses
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
>
For example, given n = 3, a solution set is:
```
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

```python
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        # Create a binary tree and read out paths
        root = Node('(') # always start with a '('
        left = n-1 # Number of left braces remaining to use
        right = n # Number of right braces remaining to use
        
        root = self.createTree(root,left,right)
        
        return self.get_paths(root)
         
    def createTree(self,node,l,r):
        if l == 0 and r == 0:
            return None
        elif l == 0:
            node.right = Node(')')
            self.createTree(node.right,0,r-1)
        elif l <= r:
            node.left = Node('(')
            self.createTree(node.left,l-1,r)
            if l < r:
                node.right = Node(')')
                self.createTree(node.right,l,r-1)
        return node
    
    def get_paths(self,node):
        if node == None:
            return []
        if node.left == None and node.right == None:
            return [node.val]
        else:
            return [node.val + val for val in self.get_paths(node.left) + self.get_paths(node.right)]

class Node(object):
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
```

## 23. Merge k Sorted Lists
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

>Example:
```
Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6
```

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        if lists == []:
            return None
        # Use heap of size k. Time complexity becomes n*O(logk) for n total elements
        # Add index to list elements to keep track of which list an element belongs to
        idx = 0
        for head in lists:
            while head != None:
                head.index = idx
                head = head.next
            idx += 1
            
        from heapq import heappush, heappop
        merged_heap = []
        
        N = len(lists)
        for idx in range(N):
            if lists[idx] != None:
                heappush(merged_heap, (lists[idx].val, lists[idx].index))
                lists[idx] = lists[idx].next
            
        head = None
        solution = None
        
        while merged_heap != []:
            popped_element = heappop(merged_heap)
            popped_value = popped_element[0]
            popped_index = popped_element[1]
            
            if lists[popped_index] != None:
                l = lists[popped_index]
                heappush(merged_heap, (l.val, l.index))
                lists[popped_index] = lists[popped_index].next
            if head == None:
                newNode = ListNode(popped_value)
                head = newNode
                solution = head
            else:
                newNode = ListNode(popped_value)
                head.next = newNode
                head = head.next
                
        return solution
```

## 26. Remove Duplicates from Sorted Array
Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this in place with constant memory.

> For example,
Given input array nums = [1,1,2], Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the new length.


```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums == []:
            return 0

        current_element = nums[0]
        scan_index = 1
        fill_index = 1

        while scan_index < len(nums):
            if nums[scan_index] != current_element:
                nums[fill_index] = nums[scan_index]
                fill_index += 1
                current_element = nums[scan_index]
            scan_index += 1

        return fill_index
```

## 27. Remove Element
Given an array and a value, remove all instances of that value in place and return the new length.

Do not allocate extra space for another array, you must do this in place with constant memory.

The order of elements can be changed. It doesn't matter what you leave beyond the new length.

> Example:
Given input array nums = [3,2,2,3], val = 3, Your function should return length = 2, with the first two elements of nums being 2.


```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        scan_index = 0 # element to scan

        L = len(nums)
        val_indices = []
        non_val_indices = []
        for i in range(L):
            if nums[i] == val:
                val_indices.append(i)
            else:
                non_val_indices.append(i)

        N = len(non_val_indices)
        if N == L or N == 0:
            return N

        # Replace the min element in val_indices with the max_element in non_val_indices and
        # continue till min element index > max_element index
        while val_indices[0] < non_val_indices[-1]:
            nums[val_indices[0]], nums[non_val_indices[-1]] = nums[non_val_indices[-1]], nums[val_indices[0]]
            val_indices = val_indices[1:]
            non_val_indices = non_val_indices[:-1]

            if len(val_indices) == 0 or len(non_val_indices) == 0:
                return N

        return N
```

## 32. Longest Valid Parentheses
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

>For "(()", the longest valid parentheses substring is "()", which has length = 2.

>Another example is ")()())", where the longest valid parentheses substring is "()()", which has length = 4.


```python
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s == "":
            return 0

        stack = []
        N = len(s)
        LVS = [0 for _ in range(N)]

        for idx in range(N):
            if stack == [] or s[idx] == '(':
                stack.append((s[idx],idx))
            elif s[idx] == ')':
                if stack[-1][0] == ')':
                    stack.append((s[idx],idx))
                else: # stack[-1] == (
                    # pop top element of stack since a valid paranthesis is formed
                    stack = stack[:-1]
                    if stack == []:
                        LVS[idx] = idx + 1
                    else:
                        LVS[idx] = idx - stack[-1][1]

        return max(LVS)
```

## 33. Search in Rotated Sorted Array
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).
>
Example 1:
```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```
>
Example 2:
```
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
```

```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        solution = self.find(nums, target)
        if solution == float('inf'):
            return -1
        else:
            return solution
        
    def find(self, nums, target):
        L = len(nums)
        if L == 0:
            return float('inf')
        if L == 1:
            if nums[0] == target:
                return 0
            else:
                return float('inf')
            
        if nums[0] == target:
            return 0
        if nums[L/2] == target:
            return L/2
        if nums[-1] == target:
            return L-1

        if nums[0] < nums[L/2]: # left half is sorted
            if nums[0] < target < nums[L/2]:
                return self.find(nums[:L/2], target)
            else:
                return L/2+self.find(nums[L/2:], target)
        else: # nums[0] > nums[L/2] # right half is sorted
            if nums[L/2] < target < nums[-1]:
                return L/2+self.find(nums[L/2:], target)
            else:
                return self.find(nums[:L/2], target)
```

## 34. Search for a Range
Given an array of integers sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

>For example,
Given [5, 7, 7, 8, 8, 10] and target value 8, return [3, 4].

```python
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if len(nums) == 0:
            return [-1,-1]
        
        first = self.findFirst(nums, target)
        if first == float('inf'):
            return [-1,-1]
        last = first + self.findLast(nums[first:], target)
        return [first, last]
        
    def findFirst(self, nums, target):
        L = len(nums)
        if L == 1:
            if nums[0] != target:
                return float('inf')
            else:
                return 0
        if nums[L/2] < target:
            if len(nums) == L/2+1:
                return float('inf')
            return L/2+1+self.findFirst(nums[L/2+1:],target)
        elif nums[L/2] > target:
            return self.findFirst(nums[:L/2], target)
        else: # nums[L/2] == target
            if L/2 == 0 or nums[L/2-1] < target:
                return L/2
            else:
                return self.findFirst(nums[:L/2], target)
            
    def findLast(self, nums, target):
        L = len(nums)
        if L == 1:
            return 0
        if nums[L/2] < target:
            return L/2+1+self.findLast(nums[L/2+1:],target)
        elif nums[L/2] > target:
            return self.findLast(nums[:L/2], target)
        else: # nums[L/2] == target
            if L/2 == len(nums)-1 or nums[L/2+1] > target:
                return L/2
            else:
                return L/2+1+self.findLast(nums[L/2+1:],target)
```

## 35. Search Insert Position
Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You may assume no duplicates in the array.
>
Example 1:
```
Input: [1,3,5,6], 5
Output: 2
```
>
Example 2:
```
Input: [1,3,5,6], 2
Output: 1
```
>
Example 3:
```
Input: [1,3,5,6], 7
Output: 4
```
>
Example 4:
```
Input: [1,3,5,6], 0
Output: 0
```

```python
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        L = len(nums)
        if nums[L/2] == target:
            return L/2
        elif L == 1:
            if nums[0] > target:
                return 0
            return 1
        elif nums[L/2] > target:
            return self.searchInsert(nums[:L/2],target)
        else:
            return L/2 + self.searchInsert(nums[L/2:],target)
```

## 36. Valid Sudoku
Determine if a 9x9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

1. Each row must contain the digits 1-9 without repetition.
2. Each column must contain the digits 1-9 without repetition.
3. Each of the 9 3x3 sub-boxes of the grid must contain the digits 1-9 without repetition.
4. The Sudoku board could be partially filled, where empty cells are filled with the character '.'.

>Example 1:
```
Input:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
Output: true
```
>Example 2:
```
Input:
[
  ["8","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being 
    modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.
```
Note:

- A Sudoku board (partially filled) could be valid but is not necessarily solvable.
- Only the filled cells need to be validated according to the mentioned rules.
- The given board contain only digits 1-9 and the character '.'.
- The given board size is always 9x9.

```python
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        if not self.check_rows(board):
            return False
        if not self.check_columns(board):
            return False
        if not self.check_boxes(board):
            return False
        return True
    
    def check_rows(self, board):
        for i in range(9):
            dictionary = {}
            for j in range(9):
                if board[i][j] == '.':
                    pass    
                elif board[i][j] not in dictionary:
                    dictionary[board[i][j]] = 1
                else:
                    return False
        return True
                
    def check_columns(self, board):
        for i in range(9):
            dictionary = {}
            for j in range(9):
                if board[j][i] == '.':
                    pass    
                elif board[j][i] not in dictionary:
                    dictionary[board[j][i]] = 1
                else:
                    return False
        return True
                
    def check_sub_boxes(self,board,I,J):
        dictionary = {}        
        for i in I:
            for j in J:
                if board[i][j] == '.':
                    pass    
                elif board[i][j] not in dictionary:
                    dictionary[board[i][j]] = 1
                else:
                    return False  
        return True
    
    def check_boxes(self,board):
        solution = True
        for (I,J) in [(range(3), range(3)), (range(3), range(3,6)), (range(3), range(6,9)), \
                      (range(3,6), range(3)), (range(3,6), range(3,6)), (range(3,6), range(6,9)), \
                      (range(6,9), range(3)), (range(6,9), range(3,6)), (range(6,9), range(6,9))]:
            solution &= self.check_sub_boxes(board,I,J)
            if solution == False:
                return False
        return True
```

## 37. Sudoku Solver
Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:

1. Each of the digits 1-9 must occur exactly once in each row.
2. Each of the digits 1-9 must occur exactly once in each column.
3. Each of the the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.

Empty cells are indicated by the character '.'.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Sudoku-by-L2G-20050714.svg/250px-Sudoku-by-L2G-20050714.svg.png)

A sudoku puzzle...

![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Sudoku-by-L2G-20050714_solution.svg/250px-Sudoku-by-L2G-20050714_solution.svg.png)

...and its solution numbers marked in red.

Note:

- The given board contain only digits 1-9 and the character '.'.
- You may assume that the given Sudoku puzzle will have a single unique solution.
- The given board size is always 9x9.

```python
class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        # Use backtracking
        # Store fixed number locations
        fixed_locations = {}
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    fixed_locations[(i,j)] = 0

        board = self.backTrack(board, (0, 0), fixed_locations)
        return
    
    def backTrack(self, board, (i, j), fixed_locations):
        # print i, j, board
        if i==8 and j==8:
            if board[i][j] == '.':
                board[i][j] = self.possibilities(board, (i, j))[0]
            return board
        if (i,j) in fixed_locations: # digit already filled
            return self.backTrack(board, self.next((i,j)), fixed_locations)
        else:
            valid_digits = self.possibilities(board, (i, j))
            for digit in valid_digits:
                board[i][j] = digit
                solved = self.backTrack(board, self.next((i,j)), fixed_locations)
                if solved != False:
                    return board
                board[i][j] = '.'
            return False
        
    def possibilities(self, board,(i, j)):
        dictionary = {}
        for k in range(9):
            if board[i][k] != '.':
                dictionary[board[i][k]] = 1
        for k in range(9):
            if board[k][j] != '.':
                dictionary[board[k][j]] = 1                
        lower1 = i/3
        lower2 = j/3
        for k in range(lower1*3,lower1*3+3):
            for l in range(lower2*3,lower2*3+3):
                if board[k][l] != '.':
                    dictionary[board[k][l]] = 1
        possibilities = []
        for i in range(1,10):
            if str(i) not in dictionary:
                possibilities += [str(i)]
        return possibilities
        
    def next(self, (i,j)):
        if j == 8:
            return (i+1,0)
        else:
            return (i,j+1)
```

## 38. Count and Say
The count-and-say sequence is the sequence of integers with the first five terms as following:

1.     1
2.     11
3.     21
4.     1211
5.     111221
1 is read off as "one 1" or 11.
11 is read off as "two 1s" or 21.
21 is read off as "one 2, then one 1" or 1211.
Given an integer n, generate the nth term of the count-and-say sequence.

Note: Each term of the sequence of integers will be represented as a string.

>Example 1:

>Input: 1
Output: "1"

>Example 2:

>Input: 4
Output: "1211"


```python
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        outputs = ['' for _ in range(n)]
        outputs[0] = '1'
        for idx in range(1,n):
            outputs[idx] = self.count_occurrences(outputs[idx-1])

        return outputs[n-1]

    def count_occurrences(self,s):
        digit = s[0]
        count = 1
        solution=''
        for idx in range(1,len(s)):
            if s[idx] == s[idx-1]:
                count += 1
            else:
                solution += str(count)
                solution += s[idx-1]
                count = 1

        solution += str(count)
        solution += s[-1]

        return solution
```

## 39. Combination Sum
Given a set of candidate numbers (C) (without duplicates) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of times.

Note:
All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
>For example, given candidate set [2, 3, 6, 7] and target 7,
A solution set is:
[
  [7],
  [2, 2, 3]
]


```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        solution = {}
        solution[0] = [[]]

        for c in candidates:
            for t in range(c,target+1):
                if t-c in solution:
                    for s in solution[t-c]:
                        if t in solution:
                            solution[t] += [s + [c]]
                        else:
                            solution[t] = [s + [c]]

        if target not in solution:
            return []
        return solution[target]
```

## 41. First Missing Positive
Given an unsorted integer array, find the first missing positive integer.

>For example,
Given [1,2,0] return 3,
and [3,4,-1,1] return 2.

Your algorithm should run in O(n) time and uses constant space.


```python
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums == []:
            return 1

        L = len(nums)
        fraction = 0.1

        for i in range(len(nums)):
            if nums[i] <=0 or nums[i] > L:
                pass
            else:
                nums[int(nums[i])-1] += fraction
        print nums

        # One final pass to determine
        for i in range(len(nums)):
            if nums[i] == int(nums[i]):
                return i+1

        return L + 1
```

## 42. Trapping Rain Water
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

>For example,
Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.

![](http://www.leetcode.com/wp-content/uploads/
2/08/rainwatertrap.png)


```python
class Solution:
    # @param {integer[]} height
    # @return {integer}
    def trap(self, height):

        if height == []:
            return 0

        area = 0
        # Find index of maximum, say idx. Then treat 0->idx a,d idx+1->end separately.
        idx = height.index(max(height))

        # Index 0 to idx. For these buildings, idx is the max height on the right.
        # Maintain max building height for buildings on left
        max_left = 0
        for i in range(0,idx):
            max_left = max(max_left,height[i])
            area += max(0,max_left-height[i])

        # Index end to idx+1. For these buildings, idx is the max height on the left.
        # Maintain max building height for buildings on right
        max_right = 0
        for i in range(len(height)-1,idx,-1):
            max_right = max(max_right,height[i])
            area += max(0,max_right-height[i])

        return area
```

## 44. Wildcard Matching
Implement wildcard pattern matching with support for '?' and '*'.

'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).

The matching should cover the entire input string (not partial).

The function prototype should be:
bool isMatch(const char *s, const char *p)

>Some examples:

>isMatch("aa","a") → false

>isMatch("aa","aa") → true

>isMatch("aaa","aa") → false

>isMatch("aa", "*") → true

>isMatch("aa", "a*") → true

>isMatch("ab", "?*") → true

>isMatch("aab", "c*a*b") → false

```python
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if s == "" and p == "*":
            return True

        # If len(p) != len(s) and p contains no '*', immediately return False
        if len(s) != len(p) and '*' not in p:
            return False

        M = len(p)
        N = len(s)
        grid = [[False for _ in range(N+1)] for _ in range(M+1)]
        grid[0][0]= True
        for i in range(1,M+1):
            for j in range(1,N+1):
                if p[i-1] == '*':
                    grid[i][j] = grid[i-1][j] or grid[i][j-1] or grid[i-1][j-1]
                elif p[i-1] == '?' or p[i-1] == s[j-1]:
                    grid[i][j] = grid[i-1][j-1]
                    k = 2
                    while k < i+1 and p[i-k] == '*':
                        grid[i][j] = grid[i][j] or grid[i-k][j-1]
                        k += 1
                else:
                    grid[i][j] = False
        return grid[M][N]
```

## 46. Permutations
Given a collection of distinct numbers, return all possible permutations.

>For example,
[1,2,3] have the following permutations:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]


```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) < 2:
            return [nums]
        permutations = self.create_permutations([],nums)
        # Permutations may be a nested list. Below is a hack to make it into one list. I need to fix it!
        import math
        while len(permutations) != math.factorial(len(nums)):
            permutations = sum(permutations,[])
        return permutations

    def create_permutations(self, current, remaining):
        if len(remaining) <= 1:
            permutations = current+remaining
        else:
            permutations = []
            for idx in range(len(remaining)):
                permutations += [self.create_permutations(current + [remaining[idx]], remaining[:idx]+remaining[idx+1:])]

        return permutations
```

## 48. Rotate Image
You are given an n x n 2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).

Follow up:
Could you do this in-place?


```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        layer = 0
        while layer < len(matrix)/2:
            self.rotate_perimeter(matrix, layer)
            print matrix
            layer += 1
        return

    def rotate_perimeter(self,matrix,layer):
        n = len(matrix)       
        tmp = matrix[0+layer][1:]
        for idx in range(layer, n-1-layer):
            matrix[0+layer][idx+1] = matrix[n-2-idx][0+layer]
        for idx in range(layer, n-1-layer):
            matrix[idx][0+layer] = matrix[n-1-layer][idx]
        for idx in range(layer, n-1-layer):
            matrix[n-1-layer][idx] = matrix[n-1-idx][n-1-layer]
        for idx in range(layer, n-1-layer):
            matrix[1+idx][n-1-layer] = tmp[idx]
        return
```

## 49. Group Anagrams
Given an array of strings, group anagrams together.

>For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
Return:
[
  ["ate", "eat","tea"],
  ["nat","tan"],
  ["bat"]
]

Note: All inputs will be in lower-case.


```python
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        dictionary = collections.defaultdict(list)
        for string in strs:
            rep = self.get_representation(string)
            dictionary[rep].append(string)
        return dictionary.values()

    def get_representation(self,string):
        rep = [0 for _ in range(26)]

        for s in string:
            idx = ord(s)-97
            rep[idx] += 1

        return tuple(rep)
```

## 53. Maximum Subarray
Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

>For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        L = len(nums)
        sums = [0 for _ in range(L)]
        sums[0] = nums[0]
        maxSum = sums[0]
        minSum = min(0,sums[0])
        for i in range(1,L):
            sums[i] += (sums[i-1]+nums[i])
            maxSum = max(maxSum,sums[i]-minSum)
            if sums[i] < minSum:
                minSum = sums[i]

        return maxSum
```

## 54. Spiral Matrix
Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

>For example,
Given the following matrix:
```
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
```
You should return [1,2,3,6,9,8,7,4,5].


```python
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        m = len(matrix)
        if m == 0:
            return matrix
        n = len(matrix[0])
        visited_indices = {}
        visited_indices[(0,0)] = 0
        solution = [matrix[0][0]]
        i = 0
        j = 0
        while len(visited_indices.keys()) < m*n:
            # Increase j
            while j < n-1:
                j += 1
                if  (i,j) not in visited_indices:
                    visited_indices[(i,j)] = 0
                    solution.append(matrix[i][j])
                else:
                    j -= 1
                    break

            # Increase i
            while i < m-1:
                i += 1
                if (i,j) not in visited_indices:
                    visited_indices[(i,j)] = 0
                    solution.append(matrix[i][j])
                else:
                    i -= 1
                    break

            # Decrease j
            while j > 0:
                j -= 1
                if (i,j) not in visited_indices:
                    visited_indices[(i,j)] = 0
                    solution.append(matrix[i][j])
                else:
                    j += 1
                    break

            # Decrease i
            while i > 0:
                i -= 1
                if (i,j) not in visited_indices:                
                    visited_indices[(i,j)] = 0
                    solution.append(matrix[i][j])
                else:
                    i += 1
                    break

        return solution
```

## 55. Jump Game
Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

>For example:
A = [2,3,1,1,4], return true.

>A = [3,2,1,0,4], return false.


```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if nums == []:
            return False

        farthest = 0
        for i in range(len(nums)):
            if i <= farthest:
                farthest = max(farthest, nums[i] + i)
                if farthest >= len(nums)-1:
                    return True

        return False
```

## 56. Merge Intervals
Given a collection of intervals, merge all overlapping intervals.

>For example,
Given [1,3],[2,6],[8,10],[15,18],
return [1,6],[8,10],[15,18].


```python
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        solution = []
        L = len(intervals)
        if L == 0:
            return solution

        # Sort intervals by start time
        intervals = sorted(intervals, key=lambda x: x.start)

        start = intervals[0].start
        final = intervals[0].end

        for i in range(1,L):
            if intervals[i].start <= final:
                final = max(final,intervals[i].end)
            else:
                solution += [(start,final)]
                start = intervals[i].start
                final = intervals[i].end

        solution += [(start,final)]
        return solution
```

## 57. Insert Interval
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

>Example 1:
Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].

>Example 2:
Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].

This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].


```python
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """

        intervals += [newInterval]
        # Sort intervals by start time
        intervals = sorted(intervals, key=lambda x: x.start)
        return self.merge(intervals)

    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        solution = []
        L = len(intervals)
        if L == 0:
            return solution

        start = intervals[0].start
        final = intervals[0].end

        for i in range(1,L):
            if intervals[i].start <= final:
                final = max(final,intervals[i].end)
            else:
                solution += [(start,final)]
                start = intervals[i].start
                final = intervals[i].end

        solution += [(start,final)]
        return solution
```

## 58. Length of Last Word
Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.

If the last word does not exist, return 0.

Note: A word is defined as a character sequence consists of non-space characters only.

>For example,
Given s = "Hello World",
return 5.


```python
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s == "":
            return 0

        letter_found = 0
        length = 0

        for i in range(len(s)-1,-1,-1):
            print i, s[i], letter_found
            if s[i] == ' ':
                if letter_found == 1:
                    return length
            else:
                letter_found = 1
                length += 1

        return length
```

## 59. Spiral Matrix II
Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.

>For example,
Given n = 3,
You should return the following matrix:
```
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]
```


```python
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        if n == 0:
            return []
        visited_indices = {}
        visited_indices[(0,0)] = 0
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        matrix[0][0] = 1
        idx = 1
        i = 0
        j = 0
        while len(visited_indices.keys()) < n*n:
            # Increase j
            while j < n-1:
                j += 1
                if  (i,j) not in visited_indices:
                    idx += 1
                    visited_indices[(i,j)] = 0
                    matrix[i][j] = idx
                else:
                    j -= 1
                    break

            # Increase i
            while i < n-1:
                i += 1
                if (i,j) not in visited_indices:
                    idx += 1
                    visited_indices[(i,j)] = 0
                    matrix[i][j] = idx
                else:
                    i -= 1
                    break

            # Decrease j
            while j > 0:
                j -= 1
                if (i,j) not in visited_indices:
                    idx += 1
                    visited_indices[(i,j)] = 0
                    matrix[i][j] = idx
                else:
                    j += 1
                    break

            # Decrease i
            while i > 0:
                i -= 1
                if (i,j) not in visited_indices:
                    idx += 1
                    visited_indices[(i,j)] = 0
                    matrix[i][j] = idx
                else:
                    i += 1
                    break

        return matrix
```

## 60. Permutation Sequence
The set [1,2,3,…,n] contains a total of n! unique permutations.

By listing and labeling all of the permutations in order,
>We get the following sequence (ie, for n = 3):
```
"123"
"132"
"213"
"231"
"312"
"321"
```

Given n and k, return the kth permutation sequence.

Note: Given n will be between 1 and 9 inclusive.


```python
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        nums = range(1,n+1)
        solution = ''
        for idx in range(n):
            nums, digit, k = self.get_next_digit(nums,k)
            solution += str(digit)

        return solution


    def get_next_digit(self,nums,k):
        n = len(nums)

        import math
        num_combinations = math.factorial(n-1)
        quotient = int((k-1)/num_combinations)
        digit = nums[quotient]
        k -= quotient*num_combinations
        nums.remove(digit)

        return nums, digit, k
```

## 62. Unique Paths
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

![](http://leetcode.com/wp-content/uploads/2014/12/robot_maze.png)

Note: m and n will be at most 100.


```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        import math
        return math.factorial(m+n-2)/(math.factorial(m-1)*math.factorial(n-1))
```

## 63. Unique Paths II
Follow up for "Unique Paths":

Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and empty space is marked as 1 and 0 respectively in the grid.

>For example,
There is one obstacle in the middle of a 3x3 grid as illustrated below.
```
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
```
The total number of unique paths is 2.

Note: m and n will be at most 100.


```python
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        num_paths = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    num_paths[i][j] = 0
                else:
                    if i == 0:
                        if j == 0:
                            num_paths[i][j] = 1 - obstacleGrid[i][j]
                        else:
                            num_paths[i][j] = num_paths[i][j-1]
                    elif j == 0:
                        num_paths[i][j] = num_paths[i-1][j]
                    else:
                        num_paths[i][j] = num_paths[i][j-1] + num_paths[i-1][j]

        return num_paths[i][j]   
```

## 64. Minimum Path Sum
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.


```python
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid)
        n = len(grid[0])
        path_sum = [[0 for _ in range(n)] for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if i == 0:
                    if j == 0:
                        path_sum[i][j] = grid[i][j]
                    else:
                        path_sum[i][j] = path_sum[i][j-1] + grid[i][j]

                elif j == 0:
                    path_sum[i][j] = path_sum[i-1][j] + grid[i][j]

                else:
                    path_sum[i][j] = min(path_sum[i][j-1],path_sum[i-1][j]) + grid[i][j]

        print path_sum
        return path_sum[i][j]
```

## 69. Sqrt(x)
Implement int sqrt(int x).

Compute and return the square root of x.

x is guaranteed to be a non-negative integer.

```python
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        # Newton-Raphson method
        r = x
        while r*r > x:
            r = (r + x/r) / 2
        return r
```

## 70. Climbing Stairs
You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Note: Given n will be a positive integer.

>
```
Example 1:
Input: 2
Output:  2
Explanation:  There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
```
>
```
Example 2:
Input: 3
Output:  3
Explanation:  There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
```

```python
class Solution:
    # @param {integer} n
    # @return {integer}
    def climbStairs(self, n):
        solution = [0]*(n+1)
        if n == 0:
            return 0
        elif n == 1:
            return 1
        elif n == 2:
            return 2
        else:            
            solution[0] = 0
            solution[1] = 1
            solution[2] = 2
            for i in range(3,n+1):
                solution[i] = solution[i-1] + solution[i-2]

        return solution[n]
```

## 72. Edit Distance
Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)

You have the following 3 operations permitted on a word:

a) Insert a character
b) Delete a character
c) Replace a character


```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        l1 = len(word1)
        l2 = len(word2)
        if l1 == 0 or l2 == 0:
            return max(l1,l2)
        if l1 == 1 and l2 == 1:
            return 1-(word1 == word2)

        edit_distance = [[0 for _ in range(l2)] for _ in range(l1)]
        found = 0 # used to avoid overcounting when same letter repeats
        for j in range(l2):
            if word2[j] == word1[0] and found == 0:
                edit_distance[0][j] = edit_distance[0][j-1]
                found = 1
            else:
                edit_distance[0][j] = edit_distance[0][j-1] + 1

        found = 0
        for i in range(l1):
            if word1[i] == word2[0] and found == 0:
                edit_distance[i][0] = edit_distance[i-1][0]
                found = 1
            else:
                edit_distance[i][0] = edit_distance[i-1][0] + 1

        for i in range(1,l1):
            for j in range(1,l2):
                if word1[i] == word2[j]:
                    edit_distance[i][j] = edit_distance[i-1][j-1]
                else:
                    edit_distance[i][j] = min(edit_distance[i][j-1], edit_distance[i-1][j], edit_distance[i-1][j-1]) + 1
        return edit_distance[i][j]
```

## 73. Set Matrix Zeroes
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.

Follow up:
Did you use extra space?
A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?


```python
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        # first convert all th rows and columns having 0s to something else, say "A". Then, do one more sweep to convert A's to 0's.
        m = len(matrix)
        n = len(matrix[0])

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    for k in range(m):
                        if matrix[k][j] != 0:
                            matrix[k][j] = 'A'
                    for l in range(n):
                        if matrix[i][l] != 0:
                            matrix[i][l] = 'A'

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 'A':
                    matrix[i][j] = 0

        return
```

## 74. Search a 2D Matrix
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.

>For example, Consider the following matrix:
```
[
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
```
Given target = 3, return true.


```python
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if matrix == [] or matrix == [[]]: # empty matrices
            return False

        m = len(matrix)
        n = len(matrix[0])
        i = 0
        j = n-1
        while i < m and j >= 0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] < target:
                i += 1
            else:
                array = matrix[i][:-1]
                return self.binarySearch(array, target)
        return False

    def binarySearch(self,array, target):
        L = len(array)
        if L == 0:
            return False
        if L == 1:
            if array[0] == target:
                return True
            else:
                return False
        median = array[L/2]
        if median == target:
            return True
        elif median < target:
            return self.binarySearch(array[L/2:], target)
        else:
            return self.binarySearch(array[:L/2], target)
```


## 75. Sort Colors
Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note:
You are not suppose to use the library's sort function for this problem.

Follow up:
A rather straight forward solution is a two-pass algorithm using counting sort.
First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array with total number of 0's, then 1's and followed by 2's.

Could you come up with an one-pass algorithm using only constant space?

```python
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """

        i = j = 0
        for k in xrange(len(nums)):
            v = nums[k]
            nums[k] = 2
            if v < 2:
                nums[j] = 1
                j += 1
            if v == 0:
                nums[i] = 0
                i += 1
```

## 79. Word Search
Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

>For example,
Given board =
```
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
```
word = "ABCCED", -> returns true,
word = "SEE", -> returns true,
word = "ABCB", -> returns false.

```python
class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        M = len(board)
        N = len(board[0])
        for i in range(M):
            for j in range(N):
                if self.search(board,i,j,word,{})[0]:
                    return True
        return False

    def search(self,board,i,j,word,visited):
        M = len(board)
        N = len(board[0])
        if (i,j) in visited:
            return None, visited
        if board[i][j] == word[0]:
            visited[(i,j)] = 0
            word = word[1:]
            if word == '':
                return True,visited
            deltas = [(0,1),(0,-1),(1,0),(-1,0)]
            for delta in deltas:
                if 0 <= i+delta[0] < M and 0 <= j+delta[1] < N:
                    solution, visited = self.search(board,i+delta[0],j+delta[1],word,visited)
                    if solution == True:
                        return True, visited
            visited.pop((i,j),None)
        return False,visited    
```


## 88. Merge Sorted Array
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

Note:
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.


```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]
```

## 91. Decode Ways
A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26
Given an encoded message containing digits, determine the total number of ways to decode it.

>For example,
Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12). The number of ways decoding "12" is 2.

```python
# Recursive solution - time limit exceeded but 222/259 cases correct. DP solution is better

class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s == '' or s[0] == "0":
            num_decodings = 0
        elif ((int(s) >= 1 and int(s) <= 10) or int(s) == 20): # Only one possible decoding for 10 and 20
            num_decodings = 1
        elif int(s) >= 11 and int(s) <= 26:
            num_decodings = 2
        else:
            num_decodings = 0
            for i in range(2): # only two digits 1-26 possible for valid input
                if (int(s[:i+1]) >= 1 and int(s[:i+1]) <= 26):
                    num_decodings += self.numDecodings(s[i+1:])

        return num_decodings

# DP solution
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s == '' or s[0] == "0":
            return 0
        num_decodings = [0]*len(s)
        num_decodings[0] = 1*(1<=int(s[0])<=9)
        if len(s) > 1:
            num_decodings[1] = num_decodings[0]*(1<=int(s[1])<=9) + 1*(10<=int(s[:2])<=26)
            for i in range(2,len(s)):
                print i
                num_decodings[i] = num_decodings[i-2]*(10<=int(s[i-1:i+1])<=26) + num_decodings[i-1]*(1<=int(s[i])<=9)

        print num_decodings   
        return num_decodings[len(s)-1]    
```

## 92. Reverse Linked List II
Reverse a linked list from position m to n. Do it in one-pass.

Note: 1 ≤ m ≤ n ≤ length of list.
>
Example:
```
Input: 1->2->3->4->5->NULL, m = 2, n = 4
Output: 1->4->3->2->5->NULL
```

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        idx = 1
        if m == 1:
            return self.reverseList(head,n-m)
        
        solution = head
        while idx < m-1:
            head = head.next
            idx += 1
            
        head.next = self.reverseList(head.next,n-m)
        return solution

    def reverseList(self, head, L):
        
        newNode = ListNode(head.val)
        tailNode = newNode
        currentNode = newNode
        
        while L > 0:
            head = head.next
            newNode = ListNode(head.val)
            newNode.next = currentNode
            currentNode = newNode
            L -= 1
            
        tailNode.next = head.next
            
        return currentNode
```

## 94. Binary Tree Inorder Traversal
Given a binary tree, return the inorder traversal of its nodes' values.

>For example:
Given binary tree [1,null,2,3],
```
   1
    \
     2
    /
   3
```
return [1,3,2].

Note: Recursive solution is trivial, could you do it iteratively?


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        stack = []
        node = root
        solution = []

        while (stack != [] or node != None):
            if node != None:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                solution.append(node.val)
                node = node.right

        return solution

```

## 96. Unique Binary Search Trees
Given n, how many structurally unique BST's (binary search trees) that store values 1...n?

>For example,
Given n = 3, there are a total of 5 unique BST's.
```
   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```


```python
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        # Catalan number
        import math
        return math.factorial(2*n)/(math.factorial(n)*math.factorial(n+1))
```

## 98. Validate Binary Search Tree
Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.

>Example 1:
```
    2
   / \
  1   3
```    
Binary tree [2,1,3], return true.

>Example 2:
```
    1
   / \
  2   3
```    
Binary tree [1,2,3], return false.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        # Scan tree in-order and make sure elements are in sorted order
        currentVal = -float("inf")
        return self.isBST(root,currentVal)[0]

    def isBST(self,root,currentVal):
        if root == None:
            return True, currentVal

        answer, currentVal = self.isBST(root.left,currentVal)
        if answer == False:
            return False, currentVal

        if root.val <= currentVal:
            return False, currentVal
        else:
            currentVal = root.val

        answer, currentVal = self.isBST(root.right,currentVal)
        if answer == False:
            return False, currentVal

        return True,currentVal
```

## 99. Recover Binary Search Tree
Two elements of a binary search tree (BST) are swapped by mistake.

Recover the tree without changing its structure.

Note:
A solution using O(n) space is pretty straight forward. Could you devise a constant space solution?


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        print self.printInOrder(root,[])

        prev_val = -float("inf")
        val1, val2, prev_val = self.inOrder(root,float("inf"),float("inf"), prev_val)
        if val2 == float("inf"): # for the case when it occurs at the very end of the traversal
            val2 = prev_val

        fixed, root = self.correctTree(root, root, val1, val2,0)
        print self.printInOrder(root,[])
        return

    def printInOrder(self,root,inOrder):
        if root == None:
            return inOrder
        inOrder = self.printInOrder(root.left,inOrder)
        inOrder += [root.val]
        inOrder = self.printInOrder(root.right,inOrder)

        return inOrder    

    def inOrder(self,root, val1, val2, prev_val):
        """
        Traverse tree in order and store the two elements that are swapped
        """
        if root == None:
            return val1, val2, prev_val
        val1, val2, prev_val = self.inOrder(root.left, val1, val2, prev_val)
        # if val1 != float("inf") and val2 != float("inf"):
        #     return val1, val2, prev_val
        val1, val2, prev_val = self.process(root.val, val1, val2, prev_val)
        # if val1 != float("inf") and val2 != float("inf"):
        #     return val1, val2, prev_val     
        val1, val2, prev_val = self.inOrder(root.right, val1, val2, prev_val)
        # if val1 != float("inf") and val2 != float("inf"):
        #     return val1, val2, prev_val

        return val1, val2, prev_val

    def process(self,current_val, val1, val2, prev_val):
        if current_val < prev_val:
            if val1 == float("inf"): # setting node 1
                val1 = prev_val
                val2 = current_val
            else: # setting node 2
                val2 = current_val
        prev_val = current_val

        return val1, val2, prev_val

    def correctTree(self,root, node, val1, val2,fixed):
        """
        Traverse tree and swap node1 and node2
        """
        if node == None:
            return fixed,root
        fixed, root = self.correctTree(root, node.left, val1, val2, fixed)
        if fixed == 1:
            return fixed,root
        fixed, node = self.swap(node,val1,val2)
        if fixed == 1:
            return fixed,root
        fixed, root = self.correctTree(root, node.right, val1, val2, fixed)
        if fixed == 1:
            return fixed,root

        return fixed,root

    def swap(self,node,val1,val2):
        """
        Swap the incorrect elements val1 and val2
        """
        fixed = 0
        if node.val == val1:
            node.val = val2
            fixed = 0
        elif node.val == val2:
            node.val = val1
            fixed = 1

        return fixed, node
```

## 100. Same Tree
Given two binary trees, write a function to check if they are equal or not.

Two binary trees are considered equal if they are structurally identical and the nodes have the same value.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p == None and q == None:
            return True
        if p == None or q == None:
            return False
        if p.val == q.val:
            return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
        return False
```

## 101. Symmetric Tree
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

>For example, this binary tree [1,2,2,3,4,4,3] is symmetric:
```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```

>But the following [1,2,2,null,3,null,3] is not:
```
    1
   / \
  2   2
   \   \
   3    3
```

Note:
Bonus points if you could solve it both recursively and iteratively.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True
        return self.isSymmetricHelper(root.left,root.right)

    def isSymmetricHelper(self,node1,node2):
        if node1 == None and node2 == None:
            return True
        if node1 == None or node2 == None:
            return False
        if node1.val != node2.val:
            return False
        return self.isSymmetricHelper(node1.left,node2.right) and self.isSymmetricHelper(node1.right,node2.left)
```

## 102. Binary Tree Level Order Traversal
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

>For example:
Given binary tree [3,9,20,null,null,15,7],
```
    3
   / \
  9  20
    /  \
   15   7
```
return its level order traversal as:
```
[
  [3],
  [9,20],
  [15,7]
]
```


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root == None:
            return []
        queue = [root]
        solution = []
        while queue != []:
            solution.append([i.val for i in queue])

            temp = []
            for i in queue:
                if i.left != None:
                    temp.append(i.left)
                if i.right != None:
                    temp.append(i.right)

            queue = temp

        return solution

```

## 103. Binary Tree Zigzag Level Order Traversal
Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

>For example:
Given binary tree [3,9,20,null,null,15,7],
```
    3
   / \
  9  20
    /  \
   15   7
```
return its zigzag level order traversal as:
```
[
  [3],
  [20,9],
  [15,7]
]
```



```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        if root == None:
            return []
        queue = [root]
        solution = []
        count = 0
        while queue != []:
            if count %2 == 0:
                solution.append([i.val for i in queue])
            else:
                solution.append([i.val for i in queue[::-1]])

            temp = []
            for i in queue:
                if i.left != None:
                    temp.append(i.left)
                if i.right != None:
                    temp.append(i.right)

            queue = temp
            count += 1

        return solution
```

## 104. Maximum Depth of Binary Tree
Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        # Use BFS
        queue = [root]
        maxDepth = 0
        while queue != []:
            tmp = queue
            queue = []
            for node in tmp:
                if node.left == None and node.right == None:
                    pass
                if node.left != None:
                    queue.append(node.left)
                if node.right != None:
                    queue.append(node.right)
            maxDepth += 1

        return maxDepth

# Using recursion
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        current_depth = 0
        max_depth = 0
        return self.mD(root,current_depth,max_depth)

    def mD(self,node,current_depth,max_depth):
        if node == None:
            return max_depth
        current_depth += 1
        max_depth = max(current_depth,max_depth)

        max_depth = self.mD(node.left,current_depth,max_depth)
        max_depth = self.mD(node.right,current_depth,max_depth)

        return max_depth    
```

## 108. Convert Sorted Array to Binary Search Tree
Given an array where elements are sorted in ascending order, convert it to a height balanced BST.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if nums == []:
            return None
        L = len(nums)
        median_index = (L-1)/2
        median = nums[median_index]

        root = TreeNode(median)
        if L > 1:
            root.left = self.sortedArrayToBST(nums[:median_index])
            root.right = self.sortedArrayToBST(nums[median_index+1:])

        return root
```

## 110. Balanced Binary Tree
Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        depths,isBalanced = self.compute_depths(root,{})
        return isBalanced


    def compute_depths(self,root,depths):
        if root == None:
            return depths,True
        depths,isBalanced = self.compute_depths(root.left,depths)
        if isBalanced == False:
            return depths, False
        depths,isBalanced = self.compute_depths(root.right,depths)
        if isBalanced == False:
            return depths, False        
        depths,isBalanced = self.process(root,depths)
        if isBalanced == False:
            return depths, False

        return depths,True

    def process(self,root,depths):
        if root.left == None and root.right == None: # leaf node
            depths[root] = 0
            return depths,True
        if root.left == None: # left child empty
            if depths[root.right] > 0:
                return depths,False
            depths[root] = depths[root.right] + 1
            return depths,True
        if root.right == None: # right child empty
            if depths[root.left] > 0:
                return depths,False
            depths[root] = depths[root.left] + 1
            return depths,True
        # Both children exist
        if abs(depths[root.left] - depths[root.right]) > 1:
            return depths,False
        depths[root] = max(depths[root.left],depths[root.right]) + 1
        return depths,True        
```

## 111. Minimum Depth of Binary Tree
Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        # Use BFS
        queue = [root]
        minDepth = 1
        while queue != []:
            tmp = queue
            queue = []
            for node in tmp:
                if node.left == None and node.right == None:
                    return minDepth
                if node.left != None:
                    queue.append(node.left)
                if node.right != None:
                    queue.append(node.right)
            minDepth += 1
```

## 112. Path Sum
Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

>For example:
Given the below binary tree and sum = 22,
```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
```        
return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        path_sums = self.getPaths(root)
        if sum in path_sums:
            return True
        return False

    def getPaths(self,root):
        if root == None:
            return []
        if root.left == None and root.right == None:
            return [root.val]
        return [root.val + val for val in self.getPaths(root.left) + self.getPaths(root.right)]
```

## 113. Path Sum II
Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

>For example:
Given the below binary tree and sum = 22,
```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
```        
return
```
[
   [5,4,11,2],
   [5,8,4,5]
]
```

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        paths = self.getPaths(root)
        desired_paths = []
        for idx in range(len(paths)):
            paths[idx] = paths[idx].split(',')
            paths[idx] = [int(val) for val in paths[idx]]
            path_sum = reduce((lambda x,y: x+y), paths[idx])
            if path_sum == sum:
                desired_paths.append(paths[idx])

        return desired_paths

    def getPaths(self,root):
        if root == None:
            return []
        if root.left == None and root.right == None:
            return [str(root.val)]
        return [str(root.val) + ',' + str(val) for val in self.getPaths(root.left) + (self.getPaths(root.right))]  
```

## 120. Triangle
Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

>For example, given the following triangle
```
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).

Note:
Bonus point if you are able to do this using only O(n) extra space, where n is the total number of rows in the triangle.


```python
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        rows = len(triangle)
        prev_total = triangle[0]
        for row in range(1,rows):
            current_total = [0 for _ in range(row+1)]
            current_total[0] = prev_total[0] + triangle[row][0]
            current_total[row] = prev_total[row-1] + triangle[row][row]
            for idx in range(1,row):
                current_total[idx] = min(prev_total[idx-1], prev_total[idx]) + triangle[row][idx]
            prev_total = current_total

        return min(prev_total)
```

## 124. Binary Tree Maximum Path Sum
Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

>Example 1:
```
Input: [1,2,3]
       1
      / \
     2   3
Output: 6
```

>Example 2:
```
Input: [-10,9,20,null,null,15,7]
   -10
   / \
  9  20
    /  \
   15   7
Output: 42
```

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # For every node in the tree, find left and right children sums separately
        return self.inOrder(root,{},-float('inf'))[1]
        
    def inOrder(self,node,pathSums,maxSum):
        if node == None:
            return pathSums,maxSum
        pathSums,maxSum = self.inOrder(node.left,pathSums,maxSum)
        pathSums,maxSum = self.inOrder(node.right,pathSums,maxSum)
        pathSums,maxSum = self.process(node,pathSums,maxSum)
        return pathSums,maxSum
        
    def process(self,node,pathSums,maxSum):
        if node.left == None and node.right == None:
            pathSums[node] = node.val
            maxSum = max(maxSum,pathSums[node])
        elif node.left == None:
            pathSums[node] = max(node.val,node.val+pathSums[node.right])
            maxSum = max(maxSum,pathSums[node]) 
        elif node.right == None:
            pathSums[node] = max(node.val,node.val+pathSums[node.left])
            maxSum = max(maxSum,pathSums[node])
        else:
            pathSums[node] = max(node.val,node.val+pathSums[node.left],node.val+pathSums[node.right])
            maxSum = max(maxSum,pathSums[node],node.val+pathSums[node.left]+pathSums[node.right])
        return pathSums,maxSum
```

## 125. Valid Palindrome
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

>For example,
"A man, a plan, a canal: Panama" is a palindrome. "race a car" is not a palindrome.

Note:
Have you consider that the string might be empty? This is a good question to ask during an interview.

For the purpose of this problem, we define empty string as valid palindrome.


```python
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = self.remove_punctuations(s)
        s = self.remove_spaces(s)
        s = self.all_lower_case(s)
        print s
        return s == s[::-1]

    def remove_punctuations(self,s):
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        return s

    def remove_spaces(self,s):
        return "".join(s.split(" "))

    def all_lower_case(self,s):
        return s.lower()
```

## 129. Sum Root to Leaf Numbers
Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.

An example is the root-to-leaf path 1->2->3 which represents the number 123.

Find the total sum of all root-to-leaf numbers.

>For example,
```
    1
   / \
  2   3
```
The root-to-leaf path 1->2 represents the number 12. The root-to-leaf path 1->3 represents the number 13. Return the sum = 12 + 13 = 25.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        paths = self.getPaths(root)
        paths = self.convert_to_int(paths)
        print paths
        return sum(int(string) for string in paths)

    def getPaths(self,root):
        if root == None:
            return []
        if root.left == None and root.right == None:
            return [str(root.val)]
        return [str(root.val) + ',' + str(val) for val in self.getPaths(root.left) + self.getPaths(root.right)]

    def convert_to_int(self,paths):
        for index in range(len(paths)):
            sum = 0
            paths[index] = paths[index].split(',')
            L = len(paths[index])
            for idx in range(L):
                sum += int(paths[index][idx])*(10**(L-1-idx))
            paths[index] = sum

        return paths
```

## 130. Surrounded Regions
Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

>For example,
```
X X X X
X O O X
X X O X
X O X X
```
After running your function, the board should be:
```
X X X X
X X X X
X X X X
X O X X
```


```python
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if board == []:
            return

        # Start from perimeter 'O' nodes. Traverse and fill connected 'O's to 'S' (say).
        m = len(board)
        n = len(board[0])
        for j in range(n):
            board = self.fill(0,j,board)
            board = self.fill(m-1,j,board)
        for i in range(m):
            board = self.fill(i,0,board)
            board = self.fill(i,n-1,board)

        # Now, convert all remaining 'O's to 'X's and finally 'S's to 'O's
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'

        for i in range(m):
            for j in range(n):
                if board[i][j] == 'S':
                    board[i][j] = 'O'

        return

    def fill(self, i,j, board):
        m = len(board)
        n = len(board[0])
        if board[i][j] == 'O':
            board[i][j] = 'S'
            deltas = [(0,1),(1,0),(-1,0),(0,-1)]
            for delta in deltas:
                if 0<=i+delta[0]<m and 0<=j+delta[1]<n:
                    board = self.fill(i+delta[0],j+delta[1],board)

        return board
```

## 135. Candy
There are N children standing in a line. Each child is assigned a rating value.

You are giving candies to these children subjected to the following requirements:

Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
What is the minimum candies you must give?

```python
class Solution:
    # @param {integer[]} ratings
    # @return {integer}
    def candy(self, ratings):
        n = len(ratings)
        candies = [1] * n

        for i in xrange(1, n):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1

        for i in xrange(n-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                candies[i] = max(candies[i], candies[i+1] + 1)

        return sum(candies)
```

## 136. Single Number
Given an array of integers, every element appears twice except for one. Find that single one.

Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?


```python
class Solution:
    # @param {integer[]} nums
    # @return {integer}
    def singleNumber(self, nums):
        answer = nums[0]
        for i in range(1,len(nums)):
            answer ^= nums[i]

        return answer
```

## 139. Word Break
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words. You may assume the dictionary does not contain duplicate words.

>For example, given
s = "leetcode",
dict = ["leet", "code"]. Return true because "leetcode" can be segmented as "leet code".


```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        if s in wordDict or len(s) == 0:
            return True

        L = len(s)

        solution = [False for _ in range(L+1)]
        for idx in range(1,L+1):
            print solution
            if s[:idx] in wordDict:
                solution[idx] = True
            else:
                for j in range(idx):
                    if solution[j] == True:
                        solution[idx] |= s[j:idx] in wordDict
                        if solution[idx] == True:
                            break


        return solution[-1]
```

## 144. Binary Tree Preorder Traversal
Given a binary tree, return the preorder traversal of its nodes' values.

>For example:
Given binary tree {1,#,2,3},
```
   1
    \
     2
    /
   3
```
return [1,2,3].

Note: Recursive solution is trivial, could you do it iteratively?


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        return self.preOrder(root,[])

    def preOrder(self,node,solution):
        if node == None:
            return solution
        solution += [node.val]
        solution = self.preOrder(node.left,solution)
        solution = self.preOrder(node.right,solution)
        return solution
```

## 153. Find Minimum in Rotated Sorted Array
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

Find the minimum element.

You may assume no duplicate exists in the array.
>
Example 1:
```
Input: [3,4,5,1,2] 
Output: 1
```
>
Example 2:
```
Input: [4,5,6,7,0,1,2]
Output: 0
```

```python
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        L = len(nums)
        if L == 1:
            return nums[0]
        if nums[0] < nums[L/2]: # left half is sorted
            if nums[-1] > nums[0]:
                return nums[0]
            else:
                return self.findMin(nums[L/2:])
        else: # nums[0] > nums[L/2] # right half is sorted
            if nums[L/2-1] > nums[L/2]:
                return nums[L/2]
            else:
                return self.findMin(nums[:L/2])
```
## 179. Largest Number
Given a list of non negative integers, arrange them such that they form the largest number.

>For example, given [3, 30, 34, 5, 9], the largest formed number is 9534330.

Note: The result may be very large, so you need to return a string instead of an integer.

```python
class Solution:
    # @param {integer[]} nums
    # @return {string}
    def largestNumber(self, nums):
        solution = ''
        if nums == []:
            return solution

        # Find largest length of int. Make all numbers equal to same length, then sort
        num_strs = map(str,nums)
        max_len = max([len(string) for string in num_strs])
        num_strs_appended = map(str,nums)

        for i in range(len(num_strs)):
            num_strs_appended[i] += num_strs[i][0]*(max_len-len(num_strs[i]))

        num_dict = {}
        for i in range(len(num_strs)):
            if num_strs_appended[i] in num_dict.keys():
                num_dict[num_strs_appended[i]] += num_strs[i]
            else:
                num_dict[num_strs_appended[i]] = num_strs[i]

        # Now sort dictionary keys
        sorted_keys = sorted([key for key in num_dict.keys()],reverse=True)

        solution = reduce((lambda x, y: x + y),[num_dict[key] for key in sorted_keys])

        if int(solution) == 0:
            solution = '0'

        return solution
```

## 191. Number of 1 Bits
Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).

For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.

```python
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        bits = 0
        while n != 0:
            bits += 1
            n &= (n-1)

        return bits
```

## 198. House Robber
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.


```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        L =len(nums)
        if L == 0:
            return 0
        rob_sum = [0 for _ in range(L)]
        rob_sum[0] = nums[0]
        if L > 1:
            rob_sum[1] = max(nums[0],nums[1])

        for i in range(2,L):
            rob_sum[i] = max(rob_sum[i-2]+nums[i],rob_sum[i-1])

        return rob_sum[L-1]
```

## 199. Binary Tree Right Side View
Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

>For example:
Given the following binary tree,
```
   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
```
You should return [1, 3, 4].


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root == None:
            return []
        current_nodes = [root]
        solution = []
        idx = 0
        while current_nodes != []:
            solution.append(current_nodes[-1].val)
            tmp = []
            for node in current_nodes:
                if node.left != None:
                    tmp.append(node.left)
                if node.right != None:
                    tmp.append(node.right)
                current_nodes = tmp

        return solution
```

## 200. Number of Islands
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

>Example 1:
```
11110
11010
11000
00000
```
Answer: 1

>Example 2:
```
11000
11000
00100
00011
```
Answer: 3


```python
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if grid == []:
            return 0

        num_islands = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    num_islands += 1
                    grid = self.traverse(i,j,grid)

        return num_islands

    def traverse(self, i, j, grid): # DFS traversal of graph
        if not (0<=i<len(grid) and 0<=j<len(grid[0])):
            return
        if grid[i][j] == "0":
            return
        else:
            grid[i][j] = "0"
            self.traverse(i-1,j,grid)
            self.traverse(i+1,j,grid)
            self.traverse(i,j+1,grid)
            self.traverse(i,j-1,grid)

        return grid
```

## 201. Bitwise AND of Numbers Range
Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.
>
For example, given the range [5, 7], you should return 4.

```python
class Solution(object):
    def rangeBitwiseAnd(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        i = 0
        while m != n:
            m >>= 1
            n >>= 1
            i += 1
        return n << i
```

## 202. Happy Number
Write an algorithm to determine if a number is "happy".

A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.

>Example: 19 is a happy number
```
1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1
```

```python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        # Use slow and fast pointers to search for loops
        sum_slow = n
        sum_fast = n
        while sum_slow != 1 and sum_fast != 1:
            sum_slow = self.digit_squared_sum(sum_slow)
            if sum_slow == 1:
                return True
            sum_fast = self.digit_squared_sum(sum_fast)
            sum_fast = self.digit_squared_sum(sum_fast)
            if sum_slow == sum_fast:
                return False
        return True

    def digit_squared_sum(self,n):
        sum = 0
        while n != 0:
            units_place = n % 10
            sum += (units_place) ** 2
            n = (n - units_place)/10

        return sum
```

## 204. Count Primes
Count the number of prime numbers less than a non-negative number, n.
>
Example:
```
Input: 10
Output: 4
Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.
```

```python
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """

        if n <= 2:
            return 0
        
        s = [1] * n
        s[0] = s[1] = 0
        for i in range(2, int(n ** 0.5) + 1):
            if s[i] == 1:
                s[i*i:n:i] = [0] * int((n-i*i-1)/i + 1)               
        return sum(s)
```

## 206. Reverse Linked List
Reverse a singly linked list.
>
Example:
```
Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
```

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        currentNode = None
        
        while head != None:
            newNode = ListNode(head.val)
            newNode.next = currentNode
            currentNode = newNode
            head = head.next
            
        return currentNode
```

## 208. Implement Trie (Prefix Tree)
Implement a trie with insert, search, and startsWith methods.

Note:
You may assume that all inputs are consist of lowercase letters a-z.


```python
class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        currentNode = self.root
        for w in word:
            if w in currentNode.children:
                currentNode = currentNode.children[w]
            else:
                newNode = TrieNode()
                newNode.val = w
                currentNode.children[w] = newNode
                currentNode = newNode
        currentNode.endOfWord = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        currentNode = self.root
        for w in word:
            if w in currentNode.children:
                currentNode = currentNode.children[w]
            else:
                return False
        if currentNode.endOfWord == True:
            return True
        else:
            return False


    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        currentNode = self.root
        for p in prefix:
            if p in currentNode.children:
                currentNode = currentNode.children[p]
            else:
                return False
        return True

class TrieNode(object):
    def __init__(self):
        self.val = None
        self.endOfWord = False
        self.children = {}

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

## 213. House Robber II
Note: This is an extension of 199. House Robber.

After robbing those houses on that street, the thief has found himself a new place for his thievery so that he will not get too much attention. This time, all houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, the security system for these houses remain the same as for those in the previous street.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.


```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        elif len(nums) == 1:
            return nums[0]
        else:
            # Compute max of rob_helper(nums[:-1] and nums[1:])
            return max(self.rob_helper(nums[:-1]),self.rob_helper(nums[1:]))

    def rob_helper(self, nums):
        L =len(nums)
        if L == 0:
            return 0
        rob_sum = [0 for _ in range(L)]
        rob_sum[0] = nums[0]
        if L > 1:
            rob_sum[1] = max(nums[0],nums[1])

        for i in range(2,L):
            rob_sum[i] = max(rob_sum[i-2]+nums[i],rob_sum[i-1])

        return rob_sum[L-1]
```

## 215. Kth Largest Element in an Array
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

>For example,
Given [3,2,1,5,6,4] and k = 2, return 5.

Note:
You may assume k is always valid, 1 ≤ k ≤ array's length.


```python
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        while k > 0:

            nums, k = self.partition_and_select(nums,k)

        return nums

    def partition_and_select(self,nums,k):
        # Quick-select algorithm
        pivot_index = 0 # chosen randomly
        pivot = nums[pivot_index]
        for i in xrange(1,len(nums)):
            if nums[i] > pivot:
                nums = [nums[i]] + nums[:i] + nums[i+1:]
                pivot_index += 1

        if pivot_index == k-1: # number found!!
            return nums[k-1], 0
        elif pivot_index < k-1:
            return nums[pivot_index+1:], k-1 - pivot_index
        else:
            return nums[:pivot_index], k

```

## 217. Contains Duplicate
Given an array of integers, find if the array contains any duplicates. Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.


```python
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        dictionary = {}
        for num in nums:
            if num in dictionary:
                return True
            else:
                dictionary[num] = 0

        return False
```

## 218. The Skyline Problem
A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance. Now suppose you are given the locations and height of all the buildings as shown on a cityscape photo (Figure A), write a program to output the skyline formed by these buildings collectively (Figure B).

![](https://leetcode.com/static/images/problemset/skyline1.jpg)![](https://leetcode.com/static/images/problemset/skyline2.jpg)

The geometric information of each building is represented by a triplet of integers [Li, Ri, Hi], where Li and Ri are the x coordinates of the left and right edge of the ith building, respectively, and Hi is its height. It is guaranteed that 0 ≤ Li, Ri ≤ INT_MAX, 0 < Hi ≤ INT_MAX, and Ri - Li > 0. You may assume all buildings are perfect rectangles grounded on an absolutely flat surface at height 0.

>For instance, the dimensions of all buildings in Figure A are recorded as: [ [2 9 10], [3 7 15], [5 12 12], [15 20 10], [19 24 8] ] .

The output is a list of "key points" (red dots in Figure B) in the format of [ [x1,y1], [x2, y2], [x3, y3], ... ] that uniquely defines a skyline. A key point is the left endpoint of a horizontal line segment. Note that the last key point, where the rightmost building ends, is merely used to mark the termination of the skyline, and always has zero height. Also, the ground in between any two adjacent buildings should be considered part of the skyline contour.

>For instance, the skyline in Figure B should be represented as:[ [2 10], [3 15], [7 12], [12 0], [15 10], [20 8], [24, 0] ].

Notes:

The number of buildings in any input list is guaranteed to be in the range [0, 10000].
The input list is already sorted in ascending order by the left x position Li.
The output list must be sorted by the x position.
There must be no consecutive horizontal lines of equal height in the output skyline. For instance, [...[2 3], [4 5], [7 5], [11 5], [12 7]...] is not acceptable; the three lines of height 5 should be merged into one in the final output as such: [...[2 3], [4 5], [12 7], ...]


```python
class Solution(object):
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """

        # Storing buildings as a dict for easy sorting
        if buildings == []:
            return []
        buildings_dict = {}
        for building in buildings:
            key = building[0]
            value = building[2]
            if key in buildings_dict.keys():
                buildings_dict[key] = max(buildings_dict[key],value)
            else:
                buildings_dict[key] = value

            key = building[1]
            value = building[2]
            if key in buildings_dict.keys():
                buildings_dict[key] = max(buildings_dict[key],value)
            else:
                buildings_dict[key] = value

        # Sort dictionary by keys
        sorted_keys = sorted(buildings_dict.keys())
        max_height = 0

        max_height_old = buildings_dict[sorted_keys[0]]
        building_heights = [max_height_old]

        solution = [[sorted_keys[0],max_height_old]]

        for key in sorted_keys[1:]:
            max_height_old = max_height
            if buildings_dict[key] in building_heights:
                building_heights.remove(buildings_dict[key]) # right edge of building
                if building_heights == []:
                    max_height = 0
                else:
                    max_height = max(building_heights)
            else:
                building_heights.append(buildings_dict[key])
                max_height = max(max_height,buildings_dict[key])

            if max_height != max_height_old:
                solution.append([key,max_height])


        return solution
```

## 219. Contains Duplicate II
Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.


```python
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        dictionary = {}
        for i in range(len(nums)):
            if nums[i] in dictionary:
                dictionary[nums[i]] += [i]
                if dictionary[nums[i]][-1] - dictionary[nums[i]][-2] <= k:
                    return True
            else:
                dictionary[nums[i]] = [i]
        return False
```

## 221. Maximal Square
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

>For example, given the following matrix:
```
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
```
Return 4.


```python
class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if matrix == []:
            return 0

        m = len(matrix)
        n = len(matrix[0])
        square_dim = [[0 for _ in range(n)] for _ in range(m)]
        max_square_dim = 0

        for j in range(n):
            if matrix[0][j] == '1':
                square_dim[0][j] = 1
                max_square_dim = max(max_square_dim,1)

        for i in range(1,m):
            if matrix[i][0] == '1':
                square_dim[i][0] = 1
                max_square_dim = max(max_square_dim,1)

        for i in range(1,m):
            for j in range(1,n):
                if matrix[i][j] == '0':
                    square_dim[i][j] = 0
                else:
                    square_dim[i][j] = 1 + min(square_dim[i][j-1],square_dim[i-1][j],square_dim[i-1][j-1])
                    max_square_dim = max(max_square_dim,square_dim[i][j])

        return max_square_dim**2
```

## 222. Count Complete Tree Nodes
Given a complete binary tree, count the number of nodes.

Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # use pre-order traversal of tree and maintain node depths. Once depth of leaf < max_depth, return and then compute
        if root == None:
            return 0
        count,depths,max_depth,_ = self.traverse(root,0,{},-1,0,0)
        return sum([2**h for h in range(max_depth)]) + count

    def traverse(self,node,count,depths,current_depth,max_depth,stopTraversing):
        if node == None:
            return count, depths, max_depth, stopTraversing

        count, depths, max_depth, stopTraversing = self.process(node,count,depths,current_depth,max_depth,stopTraversing) # process current node
        if stopTraversing == 1:
            return count, depths, max_depth, stopTraversing        

        count, depths, max_depth, stopTraversing = self.traverse(node.left,count,depths,depths[node],max_depth,stopTraversing)
        if stopTraversing == 1:
            return count, depths, max_depth, stopTraversing

        count, depths, max_depth, stopTraversing = self.traverse(node.right,count,depths,depths[node],max_depth,stopTraversing)
        if stopTraversing == 1:
            return count, depths, max_depth, stopTraversing

        return count, depths, max_depth, stopTraversing

    def process(self,node,count,depths,current_depth,max_depth,stopTraversing):
        depths[node] = current_depth + 1
        max_depth = max(max_depth, current_depth + 1)
        if node.left == None and node.right == None:
            if depths[node] < max_depth:
                stopTraversing = 1
                return count, depths, max_depth, stopTraversing
            else:
                count += 1

        return count, depths, max_depth, stopTraversing
```

## 223. Find the total area covered by two rectilinear rectangles in a 2D plane.

Each rectangle is defined by its bottom left corner and top right corner as shown in the figure.

![](https://leetcode.com/static/images/problemset/rectangle_area.png)

Assume that the total area is never beyond the maximum possible value of int.


```python
class Solution(object):
    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """
        area1 = (D-B)*(C-A)
        area2 = (H-F)*(G-E)
        if G <= A or E >= C:
            intersecting_area = 0
        elif H <= B or F >= D:
            intersecting_area = 0
        else:    
            intersecting_area = (min(C,G) - max(A,E)) * (min(H,D) - max(F,B))

        print intersecting_area, area1, area2
        return area1 + area2 - intersecting_area
```

## 226. Invert Binary Tree
Invert a binary tree.
>
```
     4
   /   \
  2     7
 / \   / \
1   3 6   9
```
to
```
     4
   /   \
  7     2
 / \   / \
9   6 3   1
```


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root == None:
            return []
        newRoot = TreeNode(root.val)
        if root.left != None:
            newRoot.right = self.invertTree(root.left)
        else:
            newRoot.right = None

        if root.right != None:
            newRoot.left = self.invertTree(root.right)
        else:
            newRoot.left = None

        return newRoot
```

## 228. Summary Ranges
Given a sorted integer array without duplicates, return the summary of its ranges.
>Example 1:
```
Input: [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
```

>Example 2:
```
Input: [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
```


```python
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        if nums == []:
            return nums

        start = nums[0]
        end = None
        current = start
        solution = []

        for num in nums[1:]:
            if num > current + 1:
                if end == None:
                    solution += [str(start)]
                else:
                    solution += [str(start)+"->"+str(end)]
                start = num
                end = None

            else:
                end = num

            current = num

        if end == None:
            solution += [str(start)]
        else:
            solution += [str(start)+"->"+str(end)]

        return solution
```

## 230. Kth Smallest Element in a BST
Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note:
You may assume k is always valid, 1 <= k <= BST's total elements.

Follow up:
What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?

**If we need to find kth smallest frequently, we might need to use a heap structure???**

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        # Perform in-order traversal and return and kth element
        return self.IOT(root,k)[0]

    def IOT(self,root,k):
        if root == None:
            return None,k
        result, k = self.IOT(root.left,k)
        if k == 0:
            return result, k

        k -= 1
        print k
        if k == 0:
            return root.val, k

        result, k = self.IOT(root.right,k)
        if k == 0:
            return result, k

        return None,k
```

## 231. Power of Two
Given an integer, write a function to determine if it is a power of two.

```python
class Solution:
    # @param {integer} n
    # @return {boolean}
    def isPowerOfTwo(self, n):
        if n <= 0:
            return False
        if n & (n-1) == 0:
            return True
        else:
            return False
```

## 235. Lowest Common Ancestor of a Binary Search Tree
Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).”
```
        _______6______
       /              \
    ___2__          ___8__
   /      \        /      \
   0      _4       7       9
         /  \
         3   5
```            
>For example, the lowest common ancestor (LCA) of nodes 2 and 8 is 6. Another example is LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if p == root or q == root or (p.val < root.val and q.val > root.val) or (p.val > root.val and q.val < root.val):
            return root
        elif p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left,p,q)
        else: # p.val, q.val > root.val
            return self.lowestCommonAncestor(root.right,p,q)
```

## 236. Lowest Common Ancestor of a Binary Tree
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).”
```
        _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2       0       8
         /  \
         7   4
```         
>For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3. Another example is LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        p_path = self.pathFromRoot(root,p)
        q_path = self.pathFromRoot(root,q)
        idx = 0
        l1 = len(p_path)
        l2 = len(q_path)
        while p_path[idx] == q_path[idx]:
            idx += 1
            if idx == l1:
                return p_path[-1]
            if idx == l2:
                return q_path[-1]
        return p_path[idx-1]

    def pathFromRoot(self,root,node):
        if root == None:
            return None
        if node == root:
            return [node]
        path = self.pathFromRoot(root.left,node)
        if path != None:
            return [root]+path
        path = self.pathFromRoot(root.right,node)
        if path != None:
            return [root]+path

        return None
```

## 240. Search a 2D Matrix II
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
>For example, consider the following matrix:
```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```

>Given target = 5, return true.

> Given target = 20, return false.


```python
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if matrix == [] or matrix == [[]]: # empty matrices
            return False

        m = len(matrix)
        n = len(matrix[0])
        i = 0
        j = n-1
        while i < m and j >= 0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] < target:
                i += 1
                print i
            else:
                j -= 1
        return False
```

## 257. Binary Tree Paths
Given a binary tree, return all root-to-leaf paths.

>For example, given the following binary tree:
```
   1
 /   \
2     3
 \
  5
```
All root-to-leaf paths are: ["1->2->5", "1->3"]


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if root == None:
            return []
        if root.left == None and root.right == None:
            return [str(root.val)]
        return [str(root.val) + '->' + str(val) for val in self.binaryTreePaths(root.left) + self.binaryTreePaths(root.right)]
```

## 263. Ugly Number
Write a program to check whether a given number is an ugly number.

Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.
>For example, 6, 8 are ugly while 14 is not ugly since it includes another prime factor 7.

Note that 1 is typically treated as an ugly number.


```python
class Solution(object):
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num <= 0:
            return False
        if num in [1,2,3,5]:
            return True
        if (num % 2) != 0 and (num % 3) != 0 and (num % 5) != 0:
            return False
        if num % 2 == 0:
            return self.isUgly(num/2)
        elif num % 3 == 0:
            return self.isUgly(num/3)
        elif num % 5 == 0:
            return self.isUgly(num/5)

        return False
```

## 264. Ugly Number II
Write a program to find the n-th ugly number.

Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.
>For example, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 is the sequence of the first 10 ugly numbers.

Note that 1 is typically treated as an ugly number, and n does not exceed 1690.


```python
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        ugly_numbers = [1]
        ugly_number = [0 for _ in range(n+1)]

        for idx in range(1,n+1)
        :
            ugly_number[idx] = min(ugly_numbers)
            if ugly_number[idx]*2 not in ugly_numbers:
                ugly_numbers.append(ugly_number[idx]*2)
            if ugly_number[idx]*3 not in ugly_numbers:
                ugly_numbers.append(ugly_number[idx]*3)
            if ugly_number[idx]*5 not in ugly_numbers:
                ugly_numbers.append(ugly_number[idx]*5)

            #print ugly_number[idx], ugly_numbers
            ugly_numbers.remove(ugly_number[idx])

        return ugly_number[idx]
```

## 268. Missing Number
Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

>For example,
Given nums = [0, 1, 3] return 2.

Note:
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?


```python
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return len(nums)*(len(nums)+1)/2 - sum(nums)
```

## 274. H-Index

According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each."

>For example, given citations = [3, 0, 6, 1, 5], which means the researcher has 5 papers in total and each of them had received 3, 0, 6, 1, 5 citations respectively. Since the researcher has 3 papers with at least 3 citations each and the remaining two with no more than 3 citations each, his h-index is 3.

Note: If there are several possible values for h, the maximum one is taken as the h-index.


```python
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """

        L = len(citations)
        citations.sort()

        for i in range(L):
            if citations[i] >= L-i:
                return L - i

        return 0
```

## 279. Perfect Squares
Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.

>For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, return 2 because 13 = 4 + 9.


```python
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0
        # Use Dynamic Programming
        import math
        num_elements = [1]
        squares = []

        for i in range(1,n+1):
            if int(math.sqrt(i)) == math.sqrt(i): # Perfect square
                num_elements.append(1)
                squares.append(i)
            else:
                num_elements.append(1 + min([num_elements[i-j] for j in squares]))

        return num_elements[-1]
```

## 287. Find the Duplicate Number
Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.

Note:
You must not modify the array (assume the array is read only).
You must use only constant, O(1) extra space.
Your runtime complexity should be less than O(n2).
There is only one duplicate number in the array, but it could be repeated more than once.


```python

```




    ['0000']



## 289. Game of Life
According to the Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

Given a board with m by n cells, each cell has an initial state live (1) or dead (0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):

- Any live cell with fewer than two live neighbors dies, as if caused by under-population.
- Any live cell with two or three live neighbors lives on to the next generation.
- Any live cell with more than three live neighbors dies, as if by over-population..
- Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
Write a function to compute the next state (after one update) of the board given its current state.

Follow up:
Could you solve it in-place? Remember that the board needs to be updated at the same time: You cannot update some cells first and then use their updated values to update other cells.
In this question, we represent the board using a 2D array. In principle, the board is infinite, which would cause problems when the active area encroaches the border of the array. How would you address these problems?

```python
class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        m = len(board)
        n = len(board[0])

        num_neighbors = [[0 for _ in range(n)] for _ in range(m)]

        for i in range(m):
            for j in range(n):
                num_neighbors[i][j] = self.num_neighbors(board,i,j)

        for i in range(m):
            for j in range(n):
                if board[i][j] == 1 and num_neighbors[i][j] < 2:
                    board[i][j] = 0

        for i in range(m):
            for j in range(n):
                if board[i][j] == 1 and num_neighbors[i][j] > 3:
                    board[i][j] = 0

        for i in range(m):
            for j in range(n):
                if board[i][j] == 0 and num_neighbors[i][j] == 3:
                    board[i][j] = 1                    

        return

    def num_neighbors(self,grid,m,n):
        sum = 0
        M = len(grid)
        N = len(grid[0])
        for i in range(max(0,m-1),min(m+1,M-1)+1):
            for j in range(max(0,n-1),min(n+1,N-1)+1):
                sum += grid[i][j]
        return sum - grid[m][n]
```

## 299. Bulls and Cows
You are playing the following Bulls and Cows game with your friend: You write down a number and ask your friend to guess what the number is. Each time your friend makes a guess, you provide a hint that indicates how many digits in said guess match your secret number exactly in both digit and position (called "bulls") and how many digits match the secret number but locate in the wrong position (called "cows"). Your friend will use successive guesses and hints to eventually derive the secret number.

>For example:
Secret number:  "1807"
Friend's guess: "7810"
Hint: 1 bull and 3 cows. (The bull is 8, the cows are 0, 1 and 7.)
Write a function to return a hint according to the secret number and friend's guess, use A to indicate the bulls and B to indicate the cows. In the above example, your function should return "1A3B".

Please note that both secret number and friend's guess may contain duplicate digits, for example:

>Secret number:  "1123"
Friend's guess: "0111"
In this case, the 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow, and your function should return "1A1B".
You may assume that the secret number and your friend's guess only contain digits, and their lengths are always equal.

```python
class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        bulls = 0
        cows = 0

        true_digit_counts = {}
        for i in range(10):
            true_digit_counts[i] = 0

        for i in range(len(secret)):
            true_digit_counts[int(secret[i])] += 1

        guess_digit_counts = {}
        for i in range(10):
            guess_digit_counts[i] = 0

        for i in range(len(secret)):
            guess_digit_counts[int(guess[i])] += 1

        for i in range(len(secret)):
            if secret[i] == guess[i]:
                guess_digit_counts[int(secret[i])] -= 1
                true_digit_counts[int(secret[i])] -= 1
                bulls += 1

        for key in guess_digit_counts.keys():
            if guess_digit_counts[key] > 0:
                if key in true_digit_counts.keys():
                    if true_digit_counts[key] > 0:
                        cows += min(guess_digit_counts[key],true_digit_counts[key])

        return str(bulls)+"A"+str(cows)+"B"
```

## 300. Longest Increasing Subsequence
Given an unsorted array of integers, find the length of longest increasing subsequence.

>For example,
Given [10, 9, 2, 5, 3, 7, 101, 18],
The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. Note that there may be more than one LIS combination, it is only necessary for you to return the length.

Your algorithm should run in O(n2) complexity.

Follow up: Could you improve it to O(n log n) time complexity?


```python
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums == []:
            return 0
        L = len(nums)
        LIS = [1 for _ in range(L)]
        for i in range(1,L):
            for j in range(i):
                if nums[i] > nums[j]:
                    LIS[i] = max(LIS[i],LIS[j]+1)
            LIS[i] = max(LIS[i],1)

        return max(LIS)
```

## 312. Burst Balloons
Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums. You are asked to burst all the balloons. If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins. Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.

Find the maximum coins you can collect by bursting the balloons wisely.

Note:
(1) You may imagine nums[-1] = nums[n] = 1. They are not real therefore you can not burst them.
(2) 0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100

>Example:
Given [3, 1, 5, 8],
Return 167
```
    nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
   coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167
```


```python
## Recursive - times out
class Solution(object):
    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        else:
            solutions = []
            for i in range(len(nums)):
                nums_minus_i = nums[:i]+nums[i+1:]
                if i-1 < 0:
                    solutions.append(self.maxCoins(nums_minus_i) + nums[i]*nums[i+1])
                elif i+1 >= len(nums):
                    solutions.append(self.maxCoins(nums_minus_i) + nums[i-1]*nums[i])
                else:
                    solutions.append(self.maxCoins(nums_minus_i) + nums[i-1]*nums[i]*nums[i+1])
            solution = max(solutions)

        return solution

## DP

```

## 313. Super Ugly Number
Write a program to find the nth super ugly number.

Super ugly numbers are positive numbers whose all prime factors are in the given prime list primes of size k.
>For example, [1, 2, 4, 7, 8, 13, 14, 16, 19, 26, 28, 32] is the sequence of the first 12 super ugly numbers given primes = [2, 7, 13, 19] of size 4.

Note:
(1) 1 is a super ugly number for any given primes.
(2) The given numbers in primes are in ascending order.
(3) 0 < k ≤ 100, 0 < n ≤ 106, 0 < primes[i] < 1000.
(4) The nth super ugly number is guaranteed to fit in a 32-bit signed integer.


```python
class Solution(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        ugly_numbers = [1]
        ugly_number = [0 for _ in range(n+1)]

        for idx in range(1,n+1):
            ugly_number[idx] = min(ugly_numbers)
            for p in primes:
                if ugly_number[idx]*p not in ugly_numbers:
                    ugly_numbers.append(ugly_number[idx]*p)

            #print ugly_number[idx], ugly_numbers
            ugly_numbers.remove(ugly_number[idx])

        return ugly_number[idx]
```

## 322. Coin Change
You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

>Example 1:
coins = [1, 2, 5], amount = 11
return 3 (11 = 5 + 5 + 1)

>Example 2:
coins = [2], amount = 3
return -1.

Note:
You may assume that you have an infinite number of each kind of coin.

```python
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount == 0:
            return 0
        # We use dynamic programming
        LARGE_NUMBER = 1e10
        num_coins = [LARGE_NUMBER]
        for i in range(1,amount+1):
            if i in coins:
                num_coins.append(1)
            else:
                min_coins = 1 + min([num_coins[max(0,i-j)] for j in coins])
                num_coins.append(min_coins)

        if num_coins[-1] >= LARGE_NUMBER:
            return -1
        else:
            return num_coins[-1]
```

## 326. Power of Three
Given an integer, write a function to determine if it is a power of three.
>
Example 1:
```
Input: 27
Output: true
```
>
Example 2:
```
Input: 0
Output: false
```
>
Example 3:
```
Input: 9
Output: true
```
>
Example 4:
```
Input: 45
Output: false
```

Follow up:
- Could you do it without using any loop / recursion?

```python
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n <= 0:
            return False
        import math
        power = math.log(n)/math.log(3)
        eps = 1e-10
        return abs(power - round(power)) < eps
```

## 344. Reverse String
Write a function that takes a string as input and returns the string reversed.

>Example:
Given s = "hello", return "olleh".


```python
class Solution(object):
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s[::-1]
```

## 357. Count Numbers with Unique Digits
Given a non-negative integer n, count all numbers with unique digits, x, where 0 ≤ x < 10n.
>
Example:
Given n = 2, return 91. (The answer should be the total numbers in the range of 0 ≤ x < 100, excluding [11,22,33,44,55,66,77,88,99])

```python
class Solution(object):
    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        # Mathematical solution
        
        count = [1 for _ in range(11)]
        
        for idx in range(1,min(10,n)+1):
            count[idx] = count[idx-1] + 9*math.factorial(9)/math.factorial(10-idx)
    
        return count[min(10,n)]
```

## 390. Elimination Game
There is a list of sorted integers from 1 to n. Starting from left to right, remove the first number and every other number afterward until you reach the end of the list.

Repeat the previous step again, but this time from right to left, remove the right most number and every other number from the remaining numbers.

We keep repeating the steps again, alternating left to right and right to left, until a single number remains.

Find the last number that remains starting with a list of length n.

>Example:
```
Input:
n = 9,
1 2 3 4 5 6 7 8 9
2 4 6 8
2 6
6
```
Output: 6


```python
class Solution(object):
    def lastRemaining(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 1:
            return 1
        n = range(2,n+1,2)
        while len(n) > 1:
            n = n[::-1]            
            n = n[1::2]
            if len(n) == 1:
                return n[0]

        return n[0]
```

## 402. Remove K Digits
Given a non-negative integer num represented as a string, remove k digits from the number so that the new number is the smallest possible.

Note:
The length of num is less than 10002 and will be ≥ k.
The given num does not contain any leading zero.

>Example 1:
```
Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
```

>Example 2:
```
Input: num = "10200", k = 1
Output: "200"
Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.
```

>Example 3:
```
Input: num = "10", k = 2
Output: "0"
Explanation: Remove all the digits from the number and it is left with nothing which is 0.
```

```python
class Solution(object):
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """

        while k > 0:
            if k >= len(num):
                return "0"
            num = self.removeDigit(num)
            k -= 1
        # Remove all the 0s in front
        while num[0] == '0' and len(num) > 1:
            num = num[1:]
        return num
    
    def removeDigit(self,num):
        # Remove the digit which is higher than its right neighbors and higher or equal to left neighbor
        current_num = num[0]
        num_left = current_num
        for idx in range(1,len(num)):
            num_right = num[idx]
            if num_left <= current_num and num_right < current_num:
                num = num[:idx-1]+num[idx:]
                return num               
            else:
                num_left = current_num
                current_num = num_right
        return num[:-1]
```

## 403. Frog Jump
A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.

Given a list of stones' positions (in units) in sorted ascending order, determine if the frog is able to cross the river by landing on the last stone. Initially, the frog is on the first stone and assume the first jump must be 1 unit.

If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units. Note that the frog can only jump in the forward direction.

Note:

- The number of stones is ≥ 2 and is < 1,100.
- Each stone's position will be a non-negative integer < 231.
- The first stone's position is always 0.
>
```
Example 1:
[0,1,3,5,6,8,12,17]
There are a total of 8 stones.
The first stone at the 0th unit, second stone at the 1st unit,
third stone at the 3rd unit, and so on...
The last stone at the 17th unit.
Return true. The frog can jump to the last stone by jumping
1 unit to the 2nd stone, then 2 units to the 3rd stone, then
2 units to the 4th stone, then 3 units to the 6th stone,
4 units to the 7th stone, and 5 units to the 8th stone.
```
>
```
Example 2:
[0,1,2,3,4,8,9,11]
Return false. There is no way to jump to the last stone as
the gap between the 5th and 6th stone is too large.
```

```python
class Solution(object):
    def canCross(self, stones):
        """
        :type stones: List[int]
        :rtype: bool
        """

        dp = {stone: {} for stone in stones}
        dp[0][0] = 0
        for stone in stones:
            for step in dp[stone].values():
                for k in [step + 1, step, step - 1]:
                    if k > 0 and stone + k in dp:
                        dp[stone + k][stone] = k
        return len(dp[stones[-1]].keys()) > 0
```

## 404. Sum of Left Leaves
Find the sum of all left leaves in a given binary tree.
>Example:
```
    3
   / \
  9  20
    /  \
   15   7
```
There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        solution = 0
        queue = [root]
        while queue != []:
            tmp = queue[0]
            if tmp.left != None:
                queue.append(tmp.left)
                if tmp.left.left == None and tmp.left.right == None:
                    solution += tmp.left.val
            if tmp.right != None:
                queue.append(tmp.right)
            queue = queue[1:]

        return solution
```

## 417. Pacific Atlantic Water Flow
Given an m x n matrix of non-negative integers representing the height of each unit cell in a continent, the "Pacific ocean" touches the left and top edges of the matrix and the "Atlantic ocean" touches the right and bottom edges.

Water can only flow in four directions (up, down, left, or right) from a cell to another one with height equal or lower.

Find the list of grid coordinates where water can flow to both the Pacific and Atlantic ocean.

Note:
The order of returned grid coordinates does not matter.
Both m and n are less than 150.
>Example: Given the following 5x5 matrix:
```
  Pacific ~   ~   ~   ~   ~
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * Atlantic
```
Return:
[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (positions with parentheses in above matrix).


```python
class Solution(object):
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if matrix == [[]] or matrix == []:
            return []
        m = len(matrix)
        n = len(matrix[0])
        print m, n

        # PACIFIC
        pacific = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            pacific[i][0] = 1
        for j in range(n):
            pacific[0][j] = 1
        for i in range(m):
            pacific = self.fill(matrix,i,0,pacific)
        for j in range(n):
            pacific = self.fill(matrix,0,j,pacific)

        #ATLANTIC
        atlantic = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            atlantic[i][n-1] = 1
        for j in range(n):
            atlantic[m-1][j] = 1
        for i in range(m):
            atlantic = self.fill(matrix,i,n-1,atlantic)
        for j in range(n):
            atlantic = self.fill(matrix,m-1,j,atlantic)

        solution = []
        for i in range(m):
            for j in range(n):
                if pacific[i][j] == 1 and atlantic[i][j] == 1:
                    solution += [[i,j]]
        return solution

    def fill(self,matrix,i,j,fill_matrix):
        m = len(matrix)
        n = len(matrix[0])

        diffs = [(0,1),(0,-1),(1,0),(-1,0)]
        for diff in diffs:
            if 0<= i+diff[0] < m and 0 <= j+diff[1] < n:
                if fill_matrix[i+diff[0]][j+diff[1]] == 0:
                    if matrix[i+diff[0]][j+diff[1]] >= matrix[i][j]:
                        fill_matrix[i+diff[0]][j+diff[1]] = 1
                        fill_matrix = self.fill(matrix,i+diff[0],j+diff[1],fill_matrix)
        return fill_matrix
```

## 434. Number of Segments in a String
Count the number of segments in a string, where a segment is defined to be a contiguous sequence of non-space characters.

Please note that the string does not contain any non-printable characters.

>Example: Input: "Hello, my name is John"
Output: 5


```python
class Solution(object):
    def countSegments(self, s):
        """
        :type s: str
        :rtype: int
        """
        return len(s.split())
```

## 442. Find All Duplicates in an Array
Given an array of integers, 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.

Find all the elements that appear twice in this array.

Could you do it without extra space and in O(n) runtime?

>
```
Example:
Input:
[4,3,2,7,8,2,3,1]
Output:
[2,3]
```

```python
class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        fraction = 0.1
        L = len(nums)
        for i in range(L):
            nums[int(nums[i])-1] += fraction

        solution = []
        for i in range(L):
            s = nums[i] - int(nums[i]) - 2*fraction
            if abs(s) < 1e-8:
                solution += [i+1]

        return solution
```

## 463. Island Perimeter
You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water. Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells). The island doesn't have "lakes" (water inside that isn't connected to the water around the island). One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

>Example:
```    
 [[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]
```
Answer: 16


```python
class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        solution = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                solution += 4-self.find_num_connections(i,j,grid)

        return solution

    def find_num_connections(self,i,j,grid):
        if grid[i][j] == 1:
            edges = 0
            if 0<=i-1<len(grid):
                edges += grid[i-1][j] == 1
            if 0<=i+1<len(grid):
                edges += grid[i+1][j] == 1
            if 0<=j-1<len(grid[0]):
                edges += grid[i][j-1] == 1
            if 0<=j+1<len(grid[0]):
                edges += grid[i][j+1] == 1
            return edges
        else:
            return 4
```

## 464. Can I Win
In the "100 game," two players take turns adding, to a running total, any integer from 1..10. The player who first causes the running total to reach or exceed 100 wins.

What if we change the game so that players cannot re-use integers?

>For example, two players might take turns drawing from a common pool of numbers of 1..15 without replacement until they reach a total >= 100.

Given an integer maxChoosableInteger and another integer desiredTotal, determine if the first player to move can force a win, assuming both players play optimally.

You can always assume that maxChoosableInteger will not be larger than 20 and desiredTotal will not be larger than 300.

>Example: Input:
maxChoosableInteger = 10
desiredTotal = 11
Output: false

Explanation:
No matter which integer the first player choose, the first player will lose.
The first player can choose an integer from 1 up to 10.
If the first player choose 1, the second player can only choose integers from 2 up to 10.
The second player will win by choosing 10 and get a total = 11, which is >= desiredTotal.
Same with other integers chosen by the first player, the second player will always win.


```python
class Solution(object):
    def canIWin(self, maxChoosableInteger, desiredTotal):
        """
        :type maxChoosableInteger: int
        :type desiredTotal: int
        :rtype: bool
        """
        return self.canFirstWin(range(1,maxChoosableInteger+1),desiredTotal,{})

    def canFirstWin(self,integerList,desiredTotal,dictionary):
        if (integerList,desiredTotal) in dictionary.keys():
            return
        else:
            dictionary[tuple(integerList+[desiredTotal])] = 0

        if desiredTotal == 0:
            return True
        if desiredTotal < 0:
            return False
        elif desiredTotal in integerList:
            return True
        else:
            for idx in range(len(integerList)):
                secondWins = self.canFirstWin(integerList[:idx]+integerList[idx+1:], desiredTotal-integerList[idx],dictionary)
                # winnable denotes if second guy can win with any integer picked by first guy
                if secondWins == False:
                    return True

        return False
```

## 485. Max Consecutive Ones
Given a binary array, find the maximum number of consecutive 1s in this array.

>Example 1:
Input: [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s.
    The maximum number of consecutive 1s is 3.


```python
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        num_consec_ones = 0
        count = 0

        for idx in range(len(nums)):
            if nums[idx] == 1:
                count += 1    
            else:
                num_consec_ones = max(count,num_consec_ones)
                count = 0        

        return max(count,num_consec_ones)
```

## 486. Predict the Winner
Given an array of scores that are non-negative integers. Player 1 picks one of the numbers from either end of the array followed by the player 2 and then player 1 and so on. Each time a player picks a number, that number will not be available for the next player. This continues until all the scores have been chosen. The player with the maximum score wins.

Given an array of scores, predict whether player 1 is the winner. You can assume each player plays to maximize his score.

>Example 1:
Input: [1, 5, 2]
Output: False

Explanation: Initially, player 1 can choose between 1 and 2.
If he chooses 2 (or 1), then player 2 can choose from 1 (or 2) and 5. If player 2 chooses 5, then player 1 will be left with 1 (or 2).
So, final score of player 1 is 1 + 2 = 3, and player 2 is 5.
Hence, player 1 will never be the winner and you need to return False.

>Example 2:
Input: [1, 5, 233, 7]
Output: True

Explanation: Player 1 first chooses 1. Then player 2 have to choose between 5 and 7. No matter which number player 2 choose, player 1 can choose 233.
Finally, player 1 has more score (234) than player 2 (12), so you need to return True representing player1 can win.
Note:
1 <= length of the array <= 20.
Any scores in the given array are non-negative integers and will not exceed 10,000,000.
If the scores of both players are equal, then player 1 is still the winner.


```python
class Solution(object):
    def PredictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        L = len(nums)
        if L == 1:
            return True

        solution = [[0 for _ in range(L)] for _ in range(L)]

        for i in range(L):
            solution[i][i] = (nums[i],0)

        for diff in range(1,L):
            for i in range(L-diff):
                j = i + diff

                if nums[i] + solution[i+1][j][1] >= nums[j] + solution[i][j-1][1]:
                    solution[i][j] = (nums[i] + solution[i+1][j][1], solution[i+1][j][0])
                else:
                    solution[i][j] = (nums[j] + solution[i][j-1][1], solution[i][j-1][0])

        return solution[i][j][0] >= solution [i][j][1]
```

## 513. Find Bottom Left Tree Value
Given a binary tree, find the leftmost value in the last row of the tree.

>Example 1:
Input:
```
    2
   / \
  1   3
```
Output:
1

>Example 2:
Input:
```
        1
       / \
      2   3
     /   / \
    4   5   6
       /
      7
```
Output:
7

Note: You may assume the tree (i.e., the given root node) is not NULL.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        queue = [root]
        while queue != []:
            tmp = []
            for q in queue:
                if q.left != None:
                    tmp.append(q.left)
                if q.right != None:
                    tmp.append(q.right)

            if tmp == []:
                return queue[0].val

            queue = tmp
```

## 515. Find Largest Value in Each Tree Row
You need to find the largest value in each row of a binary tree.
>
Example:
Input:
```
          1
         / \
        3   2
       / \   \  
      5   3   9
```
Output: [1, 3, 9]

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def largestValues(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root == None:
            return []
        queue = [root]
        solution = []
        while queue != []:
            tmp = queue
            queue = []
            maximum = tmp[0].val
            for t in tmp:
                maximum = max(maximum,t.val)
                if t.left != None:
                    queue.append(t.left)
                if t.right != None:
                    queue.append(t.right)
            solution.append(maximum)

        return solution
```

## 518. Coin Change 2
You are given coins of different denominations and a total amount of money. Write a function to compute the number of combinations that make up that amount. You may assume that you have infinite number of each kind of coin.

Note: You can assume that

0 <= amount <= 5000
1 <= coin <= 5000
the number of coins is less than 500
the answer is guaranteed to fit into signed 32-bit integer
>Example 1:
```
Input: amount = 5, coins = [1, 2, 5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
```

>Example 2:
```
Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.
```

>Example 3:
```
Input: amount = 10, coins = [10]
Output: 1
```


```python
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        num_ways = [0 for _ in range(amount+1)]
        num_ways[0] = 1
        for coin in coins:
            for idx in range(coin,len(num_ways)):
                num_ways[idx] += num_ways[idx-coin]

        return num_ways[-1]
```

## 542. 01 Matrix
Given a matrix consists of 0 and 1, find the distance of the nearest 0 for each cell.

The distance between two adjacent cells is 1.
>Example 1:
Input:
```
0 0 0
0 1 0
0 0 0
```
Output:
```
0 0 0
0 1 0
0 0 0
```

>Example 2:
Input:
```
0 0 0
0 1 0
1 1 1
```
Output:
```
0 0 0
0 1 0
1 2 1
```
Note:
The number of elements of the given matrix will not exceed 10,000.
There are at least one 0 in the given matrix.
The cells are adjacent in only four directions: up, down, left and right.


```python
class Solution(object):
    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 1:
                    matrix[i][j] = self.returnDist(i,j,matrix)

        return matrix

    # Use BFS
    def returnDist(self,i,j,matrix):
        m = len(matrix)
        n = len(matrix[0])
        dist = 0
        queue = [(i,j)]
        nodes_visited = {}
        nodes_visited[(i,j)] = 0
        while queue != []:
            temp = queue
            queue = []
            dist += 1            
            for node in temp:
                for (diff1,diff2) in [(1,0),(-1,0),(0,1),(0,-1)]:
                    if not (0<=node[0]+diff1<m and 0<=node[1]+diff2<n):
                        pass
                    else:
                        if (node[0]+diff1,node[1]+diff2) in nodes_visited:
                            pass
                        else:
                            if matrix[node[0]+diff1][node[1]+diff2] == 0:
                                return dist
                            nodes_visited[(node[0]+diff1,node[1]+diff2)] = 0
                            queue.append((node[0]+diff1,node[1]+diff2))
        return None
```

## 543. Diameter of Binary Tree
Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

>Example:
Given a binary tree
```   
          1
         / \
        2   3
       / \     
      4   5    
```
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].

Note: The length of path between two nodes is represented by the number of edges between them.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0

        max_depth = {}
        diameter = {}
        max_depth[None] = -1
        diameter[None] = -1
        max_depth, diameter = self.POT(root,max_depth,diameter)
        return max(diameter.values())

    def POT(self,node, max_depth, diameter): # Post-order traversal recording maxdepth and diameter as we traverse

        if node == None:
            pass
        else:
            self.POT(node.left, max_depth, diameter)
            self.POT(node.right, max_depth, diameter)
            max_depth[node] = max(max_depth[node.left], max_depth[node.right]) + 1
            diameter[node] = max_depth[node.left] + max_depth[node.right] + 2

        return max_depth, diameter
```

## 554. Brick Wall
There is a brick wall in front of you. The wall is rectangular and has several rows of bricks. The bricks have the same height but different width. You want to draw a vertical line from the top to the bottom and cross the least bricks.

The brick wall is represented by a list of rows. Each row is a list of integers representing the width of each brick in this row from left to right.

If your line go through the edge of a brick, then the brick is not considered as crossed. You need to find out how to draw the line to cross the least bricks and return the number of crossed bricks.

You cannot draw a line just along one of the two vertical edges of the wall, in which case the line will obviously cross no bricks.

>Example:
Input:
```
[[1,2,2,1],
 [3,1,2],
 [1,3,2],
 [2,4],
 [3,1,2],
 [1,3,1,1]]
```
Output: 2
Explanation:
![](https://leetcode.com/static/images/problemset/brick_wall.png)

Note:
The width sum of bricks in different rows are the same and won't exceed INT_MAX.
The number of bricks in each row is in range [1,10,000]. The height of wall is in range [1,10,000]. Total number of bricks of the wall won't exceed 20,000.


```python
class Solution(object):
    def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        from collections import defaultdict
        num_cuts = defaultdict(int) # dictionary to keep track of number of cuts

        if len(wall) == 1:
            if len(wall[0]) == 1:
                return 1
            else:
                return 0

        for line in wall:
            x = 0
            for j in line[:-1]:
                x += j
                num_cuts[x] += 1

        return len(wall) - max(num_cuts.values()+[0])
```

## 563. Binary Tree Tilt
Given a binary tree, return the tilt of the whole tree.

The tilt of a tree node is defined as the absolute difference between the sum of all left subtree node values and the sum of all right subtree node values. Null node has tilt 0.

The tilt of the whole tree is defined as the sum of all nodes' tilt.

>Example:
Input:
```
         1
       /   \
      2     3
```
Output: 1

Explanation:
Tilt of node 2 : 0
Tilt of node 3 : 0
Tilt of node 1 : |2-3| = 1
Tilt of binary tree : 0 + 0 + 1 = 1
Note:

The sum of node values in any subtree won't exceed the range of 32-bit integer.
All the tilt values won't exceed the range of 32-bit integer.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        tilts = {}
        tilts['none'] = 0
        sums = {}
        sums['none'] = 0
        tilts, sums = self.postOrder(root,tilts,sums)
        return sum(tilts.values())

    def postOrder(self,node,tilts,sums):
        if node == None:
            return tilts, sums            
        tilts,sums = self.postOrder(node.left,tilts,sums)
        tilts,sums = self.postOrder(node.right,tilts,sums)
        tilts,sums = self.process(node,tilts,sums)
        return tilts, sums

    def process(self,node,tilts,sums):
        if node.left == None and node.right == None:
            tilts[node] = 0
            sums[node] = node.val
        elif node.left == None:
            sums[node] = node.val + sums[node.right]
            tilts[node] = abs(-sums[node.right])
        elif node.right == None:
            sums[node] = node.val + sums[node.left]
            tilts[node] = abs(sums[node.left])
        else:
            sums[node] = node.val + sums[node.left] + sums[node.right]
            tilts[node] = abs(sums[node.left] - sums[node.right])

        return tilts,sums
```

## 605. Can Place Flowers
Suppose you have a long flowerbed in which some of the plots are planted and some are not. However, flowers cannot be planted in adjacent plots - they would compete for water and both would die.

Given a flowerbed (represented as an array containing 0 and 1, where 0 means empty and 1 means not empty), and a number n, return if n new flowers can be planted in it without violating the no-adjacent-flowers rule.

>Example 1:
Input: flowerbed = [1,0,0,0,1], n = 1
Output: True

>Example 2:
Input: flowerbed = [1,0,0,0,1], n = 2
Output: False

Note:

The input array won't violate no-adjacent-flowers rule.
The input array size is in the range of [1, 20000].
n is a non-negative integer which won't exceed the input array size.


```python
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        if n == 0:
            return True
        flowerbed = [0] + flowerbed + [0]
        for i in range(1,len(flowerbed)-1):
            if flowerbed[i-1] == 0 and flowerbed[i] == 0 and flowerbed[i+1] == 0:
                flowerbed[i] = 1
                n -= 1
            if n == 0:
                return True
        return False
```

## 628. Maximum Product of Three Numbers
Given an integer array, find three numbers whose product is maximum and output the maximum product.

>Example 1:
Input: [1,2,3]
Output: 6

>Example 2:
Input: [1,2,3,4]
Output: 24


```python
class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # Two largest negative numbers with highest positive or three largest positive numbers.

        nums.sort()

        possibility1 = nums[0]*nums[1]*nums[-1]
        possibility2 = nums[-1]*nums[-2]*nums[-3]

        return max(possibility1, possibility2)
```

## 640. Solve the Equation
Solve a given equation and return the value of x in the form of string "x=#value". The equation contains only '+', '-' operation, the variable x and its coefficient.

If there is no solution for the equation, return "No solution".

If there are infinite solutions for the equation, return "Infinite solutions".

If there is exactly one solution for the equation, we ensure that the value of x is an integer.
>
Example 1:
```
Input: "x+5-3+x=6+x-2"
Output: "x=2"
```
>
Example 2:
```
Input: "x=x"
Output: "Infinite solutions"
```
>
Example 3:
```
Input: "2x=x"
Output: "x=0"
```
>
Example 4:
```
Input: "2x+3x-6x=x+2"
Output: "x=-1"
```
>Example 5:
```
Input: "x=x+2"
Output: "No solution"
```

```python
class Solution(object):
    def solveEquation(self, equation):
        """
        :type equation: str
        :rtype: str
        """
        L = len(equation)
        
        eqn_left, eqn_right = equation.split("=")
        
        (a,b) = self.get_coeffs(eqn_left)
        (c,d) = self.get_coeffs(eqn_right)
        x_coeff = a-c
        c_coeff = b-d
        print x_coeff, c_coeff
        if x_coeff == 0 and c_coeff == 0:
            return "Infinite solutions"
        elif x_coeff == 0:
            return "No solution"
        else:
            return "x={}".format(-c_coeff/x_coeff)
        
    def get_coeffs(self, equation):
        """
        Obtain the coefficient for x and the constant coeff.
        """
        x_coeff = 0 # coefficient for x
        c_coeff = 0 # coefficient for the constant
        coeff = None
        multiplier = 1        
        
        L = len(equation)
        for i in range(L):
            if equation[i] == '-':
                if i > 0:
                    if coeff == None:
                        coeff = multiplier
                    if equation[i-1] == 'x':
                        x_coeff += coeff
                    else:
                        c_coeff += coeff
                    coeff = None
                multiplier = -1
            elif equation[i] == '+':
                if i > 0:
                    if coeff == None:
                        coeff = multiplier                    
                    if equation[i-1] == 'x':
                        x_coeff += coeff
                    else:
                        c_coeff += coeff
                    coeff = None
                multiplier = 1
            elif equation[i] != 'x': # represents an integer
                if coeff == None:
                    coeff = 0                
                coeff *= 10
                coeff += multiplier*int(equation[i])
                
        if coeff == None:
            coeff = multiplier
        if equation[i] == 'x':
            x_coeff += coeff
        else:
            c_coeff += coeff
        coeff = None
                
        return x_coeff, c_coeff
```
## 648. Replace Words
In English, we have a concept called root, which can be followed by some other words to form another longer word - let's call this word successor. For example, the root an, followed by other, which can form another word another.

Now, given a dictionary consisting of many roots and a sentence. You need to replace all the successor in the sentence with the root forming it. If a successor has many roots can form it, replace it with the root with the shortest length.

You need to output the sentence after the replacement.
>
Example 1:
```
Input: dict = ["cat", "bat", "rat"]
sentence = "the cattle was rattled by the battery"
Output: "the cat was rat by the bat"
```

Note:
- The input will only have lower-case letters.
- 1 <= dict words number <= 1000
- 1 <= sentence words number <= 1000
- 1 <= root length <= 100
- 1 <= sentence words length <= 1000

```python
class Solution(object):
    def replaceWords(self, dict, sentence):
        """
        :type dict: List[str]
        :type sentence: str
        :rtype: str
        """
        t = Trie()
        for d in dict:
            t.insert(d)
        
        solution = ""
        words = sentence.split(" ")
        for word in words:
            solution += t.find_root(word)
            solution += " "
        return solution[:-1]
        
class Trie(object):
    def __init__(self):
        self.head = TrieNode("head")
        
    def insert(self,root):
        currentNode = self.head
        L = len(root)
        for i in range(L):
            if root[i] not in currentNode.children:
                currentNode.children[root[i]] = TrieNode(root[i])
                currentNode = currentNode.children[root[i]]
            else:
                currentNode = currentNode.children[root[i]]
        currentNode.done = True
        
    def find_root(self,s):
        currentNode = self.head
        L = len(s)
        for i in range(L):
            if s[i] in currentNode.children:
                currentNode = currentNode.children[s[i]]
                if currentNode.done == True:
                    return s[:i+1]
            else:
                return s
        return s
        
class TrieNode(object):
    def __init__(self,x):
        self.val = x
        self.done = False
        self.children = {}
```

## 662.  Maximum Width of Binary Tree
Given a binary tree, write a function to get the maximum width of the given tree. The width of a tree is the maximum width among all levels. The binary tree has the same structure as a full binary tree, but some nodes are null.

The width of one level is defined as the length between the end-nodes (the leftmost and right most non-null nodes in the level, where the null nodes between the end-nodes are also counted into the length calculation.

>Example 1:
Input:
```
           1
         /   \
        3     2
       / \     \  
      5   3     9
```
Output: 4
Explanation: The maximum width existing in the third level with the length 4 (5,3,null,9).

>Example 2:
Input:
```
          1
         /  
        3    
       / \       
      5   3     
```
Output: 2
Explanation: The maximum width existing in the third level with the length 2 (5,3).

>Example 3:
Input:
```
          1
         / \
        3   2
       /        
      5      
```
Output: 2
Explanation: The maximum width existing in the second level with the length 2 (3,2).

>Example 4:
Input:
```
          1
         / \
        3   2
       /     \  
      5       9
     /         \
    6           7
```
Output: 8
Explanation:The maximum width existing in the fourth level with the length 8 (6,null,null,null,null,null,null,7).

Note: Answer will in the range of 32-bit signed integer.


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def widthOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        max_width = 0
        
        queue = [(root,0)]
        
        while queue != []:
            max_width = max(max_width,queue[-1][1]-queue[0][1]+1)            
            tmp = queue
            queue = []
            for t in tmp:
                if t[0].left != None:
                    queue.append((t[0].left,2*t[1]))
                if t[0].right != None:
                    queue.append((t[0].right,2*t[1]+1))

        return max_width
```

## 677. Map Sum Pairs
Implement a MapSum class with insert, and sum methods.

For the method insert, you'll be given a pair of (string, integer). The string represents the key and the integer represents the value. If the key already existed, then the original key-value pair will be overridden to the new one.

For the method sum, you'll be given a string representing the prefix, and you need to return the sum of all the pairs' value whose key starts with the prefix.
>Example 1:
```
Input: insert("apple", 3), Output: Null
Input: sum("ap"), Output: 3
Input: insert("app", 2), Output: Null
Input: sum("ap"), Output: 5
```


```python
class MapSum(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode(0)
        self.keys = {}

    def insert(self, key, val):
        """
        :type key: str
        :type val: int
        :rtype: void
        """
        if key in self.keys:
            add = 0
        else:
            self.keys[key] = 0
            add = 1

        currentNode = self.root
        while key != '':
            if currentNode.children == {} or key[0] not in currentNode.children:
                newNode = TrieNode(val)
                currentNode.children[key[0]] = newNode
                currentNode = newNode
            else:
                currentNode = currentNode.children[key[0]]
                if add:
                    currentNode.val += val
                else:
                    currentNode.val = val
            key = key[1:]

    def sum(self, prefix):
        """        
        :type prefix: str
        :rtype: int
        """
        currentNode = self.root
        while prefix != '':
            if prefix[0] not in currentNode.children:
                return 0
            else:
                currentNode = currentNode.children[prefix[0]]
            prefix = prefix[1:]
        return currentNode.val

class TrieNode(object):
    def __init__(self,val):
        self.children = {}
        self.val = val       

# Your MapSum object will be instantiated and called as such:
# obj = MapSum()
# obj.insert(key,val)
# param_2 = obj.sum(prefix)
```

## 692. Top K Frequent Words
Given a non-empty list of words, return the k most frequent elements.

Your answer should be sorted by frequency from highest to lowest. If two words have the same frequency, then the word with the lower alphabetical order comes first.

>Example 1:
```
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.
    Note that "i" comes before "love" due to a lower alphabetical order.
```    
>Example 2:
```
Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words,
    with the number of occurrence being 4, 3, 2 and 1 respectively.
```

Note:
- You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
- Input words contain only lowercase letters.

Follow up:
- Try to solve it in O(n log k) time and O(n) extra space.

```python
class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        
        from collections import defaultdict
        words_dict = defaultdict(int)
        for word in words:
                words_dict[word] += 1
                    
        max_len = 0
        for words in words_dict:
            max_len = max(max_len, len(words))
                
        f = lambda x: (words_complement_dict[x], x)

        # Using string complement so that we can directly use min heap.
        # This reverses the lexicographic ordering of the words, so its straightforward to use the min heap.
        def str_complement(s, pad = 1):
            sc = ''
            for char in s:
                sc += chr(ord("z")-ord(char)+ord("a"))
            if pad:
                sc += '{'*(max_len-len(s))
            return sc
        
        words_complement_dict = {}
        for word in words_dict:
            words_complement_dict[str_complement(word)] = words_dict[word]
        
        # Use heap data structure to maintain the k most frequent elements
        from heapq import heappush, heappop
        heap = []
        count = 0
        
        for w in words_complement_dict:
            count += 1
            if count <=k:
                heappush(heap,f(w))
            else:
                head = heap[0]
                if words_complement_dict[w] > head[0] or (words_complement_dict[w] == head[0] and w > head[1]):
                    heappop(heap)
                    heappush(heap,f(w))
        
        solution = []
        for _ in range(k):
            solution += [(str_complement(heappop(heap)[1].strip('{'), pad = 0))]
        
        return solution[::-1]
```

## 695. Max Area of Island
Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)
>Example 1:
```
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
 ```
Given the above grid, return 6. Note the answer is not 11, because the island must be connected 4-directionally.
```
Example 2:
[[0,0,0,0,0,0,0,0]]
```
Given the above grid, return 0.

Note: The length of each dimension in the given grid does not exceed 50.


```python
class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid)
        n = len(grid[0])
        max_area = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    grid[i][j] = 0
                    area = 1
                    grid, area = self.traverse(grid,area,i,j)
                    max_area = max(max_area, area)
        return max_area

    def traverse(self,grid,area,i,j):
        m = len(grid)
        n = len(grid[0])        
        deltas = [(0,1),(1,0),(-1,0),(0,-1)]
        for delta in deltas:
            if 0 <= i + delta[0] < m and 0 <=j + delta[1] < n:
                if grid[i + delta[0]][j+delta[1]] == 1:
                    grid[i+delta[0]][j+delta[1]] = 0
                    area += 1
                    grid, area = self.traverse(grid,area,i+delta[0],j+delta[1])

        return grid, area
```

## 728. Self Dividing Numbers
A self-dividing number is a number that is divisible by every digit it contains.

>For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.

Also, a self-dividing number is not allowed to contain the digit zero.

Given a lower and upper number bound, output a list of every possible self dividing number, including the bounds if possible.
>Example 1: Input: left = 1, right = 22. Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]

Note:
The boundaries of each input argument are 1 <= left <= right <= 10000.


```python
class Solution(object):
    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        return [n for n in range(left,right+1) if self.self_dividing(n)]

    def self_dividing(self,num):        
        return False if '0' in str(num) else all([num % int(s) == 0 for s in str(num)])
```

## 729. My Calendar I
Implement a MyCalendar class to store your events. A new event can be added if adding the event will not cause a double booking.

Your class will have the method, book(int start, int end). Formally, this represents a booking on the half open interval [start, end), the range of real numbers x such that start <= x < end.

A double booking happens when two events have some non-empty intersection (ie., there is some time that is common to both events.)

For each call to the method MyCalendar.book, return true if the event can be added to the calendar successfully without causing a double booking. Otherwise, return false and do not add the event to the calendar.

Your class will be called like this: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)
>Example 1:
```
MyCalendar();
MyCalendar.book(10, 20); // returns true
MyCalendar.book(15, 25); // returns false
MyCalendar.book(20, 30); // returns true
```

Explanation:
The first event can be booked.  The second can't because time 15 is already booked by another event.
The third event can be booked, as the first event takes every time less than 20, but not including 20.
Note:

The number of calls to MyCalendar.book per test case will be at most 1000.
In calls to MyCalendar.book(start, end), start and end are integers in the range [0, 10^9].


```python
class MyCalendar(object):

    def __init__(self):
        self.root = None

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        if self.root == None:
            self.root = Node(start,end)
            return True
        else:
            newNode = Node(start,end)
            return self.root.insert(newNode)

class Node(object):

    def __init__(self,start,final):
        self.start = start
        self.final = final
        self.left = None
        self.right = None

    def insert(self,newNode):
        if newNode.start >= self.final:
            if self.right == None:
                self.right = newNode
                return True
            else:
                return self.right.insert(newNode)
        elif newNode.final <= self.start:
            if self.left == None:
                self.left = newNode
                return True
            else:
                return self.left.insert(newNode)
        else:
            return False

# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)
```

## 733. Flood Fill
An image is represented by a 2-D array of integers, each integer representing the pixel value of the image (from 0 to 65535).

Given a coordinate (sr, sc) representing the starting pixel (row and column) of the flood fill, and a pixel value newColor, "flood fill" the image.

To perform a "flood fill", consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color as the starting pixel), and so on. Replace the color of all of the aforementioned pixels with the newColor.

At the end, return the modified image.
>Example 1:
```
Input:
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
```

Explanation:
From the center of the image (with position (sr, sc) = (1, 1)), all pixels connected
by a path of the same color as the starting pixel are colored with the new color.
Note the bottom corner is not colored 2, because it is not 4-directionally connected
to the starting pixel.
Note:

The length of image and image[0] will be in the range [1, 50].
The given starting pixel will satisfy 0 <= sr < image.length and 0 <= sc < image[0].length.
The value of each color in image[i][j] and newColor will be an integer in [0, 65535].


```python
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        oldColor = image[sr][sc]
        if oldColor == newColor:
            return image
        else:
             return self.fill(image,sr,sc,oldColor,newColor)

    def fill(self,image,sr,sc,oldColor,newColor):
        if image[sr][sc] != oldColor:
            return image

        image[sr][sc] = newColor
        m = len(image)
        n = len(image[0])
        deltas = [(0,1),(1,0),(-1,0),(0,-1)]
        for delta in deltas:
            if 0 <= sr+delta[0] < m and 0 <= sc+delta[1] < n:
                image = self.fill(image,sr+delta[0],sc+delta[1],oldColor,newColor)

        return image
```

## 735. Asteroid Collision
We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.
>Example 1:
```
Input:
asteroids = [5, 10, -5]
Output: [5, 10]
```

Explanation:
The 10 and -5 collide resulting in 10.  The 5 and 10 never collide.

>Example 2:
```
Input:
asteroids = [8, -8]
Output: []
```

Explanation:
The 8 and -8 collide exploding each other.

>Example 3:
```
Input:
asteroids = [10, 2, -5]
Output: [10]
```

Explanation:
The 2 and -5 collide resulting in -5.  The 10 and -5 collide resulting in 10.

>Example 4:
```
Input:
asteroids = [-2, -1, 1, 2]
Output: [-2, -1, 1, 2]
```

Explanation:
The -2 and -1 are moving left, while the 1 and 2 are moving right.
Asteroids moving the same direction never meet, so no asteroids will meet each other.

Note:
The length of asteroids will be at most 10000.
Each asteroid will be a non-zero integer in the range [-1000, 1000].

```python
class Solution(object):
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        if asteroids == []:
            return asteroids

        solution = []
        # Use stack
        L = len(asteroids)
        for i in range(L):
            solution = self.asteroid_traverse(solution,asteroids[i])

        return solution

    def asteroid_traverse(self,stack,asteroid):
        if stack and stack[-1] > 0 and asteroid < 0:
            if stack[-1] == -asteroid:
                return stack[:-1]
            elif stack[-1] > -asteroid:
                return stack
            else:
                return self.asteroid_traverse(stack[:-1],asteroid)
        else:
            stack += [asteroid]
            return stack
```

## 739. Daily Temperatures
Given a list of daily temperatures, produce a list that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.

>For example, given the list temperatures = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].

Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].


```python
class Solution(object):
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        # Idea : Use stack.
        L = len(temperatures)
        solution = [0 for _ in range(L)]
        
        stack = [(0,temperatures[0])]
        for i in range(1,L):
            while len(stack) > 0 and temperatures[i] > stack[-1][1]:
                solution[stack[-1][0]] = i - stack[-1][0]
                stack = stack[:-1]
            stack += [(i,temperatures[i])]
                
        return solution
```

## 743. Network Delay Time
There are N network nodes, labelled 1 to N.

Given times, a list of travel times as directed edges times[i] = (u, v, w), where u is the source node, v is the target node, and w is the time it takes for a signal to travel from source to target.

Now, we send a signal from a certain node K. How long will it take for all nodes to receive the signal? If it is impossible, return -1.

Note:
- N will be in the range [1, 100].
- K will be in the range [1, N].
- The length of times will be in the range [1, 6000].
- All edges times[i] = (u, v, w) will have 1 <= u, v <= N and 1 <= w <= 100.

```python
class Solution(object):
    def networkDelayTime(self, times, N, K):
        """
        :type times: List[List[int]]
        :type N: int
        :type K: int
        :rtype: int
        """
        from collections import defaultdict
        from heapq import heappush, heappop
        
        visited = {}
        travel_times = defaultdict(lambda: float('inf'))
        current_node = K
        current_time = 0
        visited[current_node] = 0
        travel_times[current_node] = 0
        
        # Convert times to a dictionary
        times_dict = defaultdict(list)        
        for t in times:
            times_dict[t[0]] += [(t[1],t[2])]

        heap = []
        heappush(heap,(0,K))
            
        while heap != []:
            current_time, current_node = heappop(heap)
            
            for t in times_dict[current_node]:
                if t[0] not in visited:
                    heappush(heap,(current_time+t[1],t[0]))
                    travel_times[t[0]] = min(current_time+t[1], travel_times[t[0]])
            visited[current_node] = 0


        print travel_times
        if len(visited) < N:
            return -1
        else:
            return max(travel_times.values())
```

## 746. Min Cost Climbing Stairs
On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).

Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.

>Example 1:
```
Input: cost = [10, 15, 20]
Output: 15
```
Explanation: Cheapest is start on cost[1], pay that cost and go to the top.

>Example 2:
```
Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
Output: 6
```
Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].
Note:
cost will have a length in the range [2, 1000].
Every cost[i] will be an integer in the range [0, 999].

```python
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        L = len(cost)
        minCost = [0 for _ in range(L+1)] # min cost to get to a step

        for idx in range(2,L+1):
            minCost[idx] = min(minCost[idx-1]+cost[idx-1],minCost[idx-2]+cost[idx-2])

        return minCost[-1]
```

## 747. Largest Number At Least Twice of Others
In a given integer array nums, there is always exactly one largest element.

Find whether the largest element in the array is at least twice as much as every other number in the array.

If it is, return the index of the largest element, otherwise return -1.
>
Example 1:
```
Input: nums = [3, 6, 1, 0]
Output: 1
Explanation: 6 is the largest integer, and for every other number in the array x,
6 is more than twice as big as x.  The index of value 6 is 1, so we return 1.
```
>
Example 2:
```
Input: nums = [1, 2, 3, 4]
Output: -1
Explanation: 4 isn't at least as big as twice the value of 3, so we return -1.
```

Note:

- nums will have a length in the range [1, 50].
- Every nums[i] will be an integer in the range [0, 99].

```python
class Solution(object):
    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return 0
        
        first = nums[0]
        first_index = 0
        if nums[1] > nums[0]:
            first = nums[1]
            first_index = 1
            second = nums[0]
            second_index = 0
        else:
            second = nums[1]
            second_index = 1
            
        for idx in range(2,len(nums)):
            if nums[idx]>=first:
                second = first
                second_index = first_index
                first = nums[idx]
                first_index = idx
            elif nums[idx] > second:
                second = nums[idx]
                second_index = idx
                
        if first >= second*2:
            return first_index
        else:
            return -1
```

## 752. Open the Lock
You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'. Each move consists of turning one wheel one slot.

The lock initially starts at '0000', a string representing the state of the 4 wheels.

You are given a list of deadends dead ends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.

Given a target representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.
>Example 1:
```
Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Output: 6
```

Explanation:
A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".
Note that a sequence like "0000" -> "0001" -> "0002" -> "0102" -> "0202" would be invalid,
because the wheels of the lock become stuck after the display becomes the dead end "0102".

>Example 2:
```
Input: deadends = ["8888"], target = "0009"
Output: 1
```

Explanation:
We can turn the last wheel in reverse to move from "0000" -> "0009".

>Example 3:
```
Input: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
Output: -1
```

Explanation:
We can't reach the target without getting stuck.

>Example 4:
```
Input: deadends = ["0000"], target = "8888"
Output: -1
```

Note:
The length of deadends will be in the range [1, 500].
target will not be in the list deadends.
Every string in deadends and the string target will be a string of 4 digits from the 10,000 possibilities '0000' to '9999'.

```python
class Solution(object):
    def openLock(self, deadends, target):
        """
        :type deadends: List[str]
        :type target: str
        :rtype: int
        """
        # Use BFS
        queue = ['0000']
        distance = -1
        visited = {}
        for d in deadends:
            visited[d] = 1

        while queue != []:
            distance += 1
            tmp = queue
            queue = []
            for q in tmp:
                if q == target:
                    return distance
                if q not in visited:
                    queue += self.neighbors(q)
                    visited[q] = 1
        return -1

    def neighbors(self,position):
        neighbors = []
        for idx in range(4):
            neighbors.append(position[:idx]+str((int(position[idx])+1)%10)+position[idx+1:])
            neighbors.append(position[:idx]+str((int(position[idx])-1)%10)+position[idx+1:])
        return neighbors
```

## 763. Partition Labels
A string S of lowercase letters is given. We want to partition this string into as many parts as possible so that each letter appears in at most one part, and return a list of integers representing the size of these parts.
>
Example 1:
```
Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8]
```
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.

Note:
1. S will have length in range [1, 500].
2. S will consist of lowercase letters ('a' to 'z') only.

```python
class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        # This is equivalent to non-overlapping intervals problem
        # Create interval lists for each letter denoting start and end positions
        intervals = {}
        L = len(S)
        for idx in range(L):
            if S[idx] in intervals:
                intervals[S[idx]][-1] = idx
            else:
                intervals[S[idx]] = [idx,idx]

        # Create a list of intervals
        intervals_list = sorted(intervals.values(),key=lambda x:x[0])

        # Create non-overlapping intervals
        NOI = []
        first = intervals_list[0][0]
        last = intervals_list[0][1]
        for i in intervals_list[1:]:
            if i[0] > last:
                NOI += [[first,last]]
                first = i[0]
                last = i[1]
            else:
                last = max(last,i[1])
        NOI += [[first,last]]

        return [noi[1] - noi[0] + 1 for noi in NOI]
```

## 764. Largest Plus Sign
In a 2D grid from (0, 0) to (N-1, N-1), every cell contains a 1, except those cells in the given list mines which are 0. What is the largest axis-aligned plus sign of 1s contained in the grid? Return the order of the plus sign. If there is none, return 0.

An "axis-aligned plus sign of 1s of order k" has some center grid[x][y] = 1 along with 4 arms of length k-1 going up, down, left, and right, and made of 1s. This is demonstrated in the diagrams below. Note that there could be 0s or 1s beyond the arms of the plus sign, only the relevant area of the plus sign is checked for 1s.
>
Examples of Axis-Aligned Plus Signs of Order k:
```
Order 1:
000
010
000
Order 2:
00000
00100
01110
00100
00000
Order 3:
0000000
0001000
0001000
0111110
0001000
0001000
0000000
```
>
Example 1:
```
Input: N = 5, mines = [[4, 2]]
Output: 2
Explanation:
11111
11111
11111
11111
11011
In the above grid, the largest plus sign can only be order 2.  One of them is marked in bold.
```
>
Example 2:
```
Input: N = 2, mines = []
Output: 1
Explanation:
There is no plus sign of order 2, but there is of order 1.
```
>Example 3:
```
Input: N = 1, mines = [[0, 0]]
Output: 0
Explanation:
There is no plus sign, so return 0.
```

Note:

- N will be an integer in the range [1, 500].
- mines will have length at most 5000.
- mines[i] will be length 2 and consist of integers in the range [0, N-1].
- (Additionally, programs submitted in C, C++, or C# will be judged with a slightly smaller time limit.)

```python
class Solution(object):
    def orderOfLargestPlusSign(self, N, mines):
        """
        :type N: int
        :type mines: List[List[int]]
        :rtype: int
        """
        if len(mines) == N**2: # No 1s in the grid
            return 0
        
        from collections import defaultdict
        
        grid = defaultdict(lambda: 1)
        for m in mines:
            grid[tuple(m)] = 0
        
        left_counts = defaultdict(int)
        right_counts = defaultdict(int)
        top_counts = defaultdict(int)
        bottom_counts = defaultdict(int)
        
        for i in range(1,N):
            for j in range(N):
                top_counts[(i,j)] = (top_counts[(i-1,j)]+grid[(i-1,j)]) * int(grid[(i,j)] == 1)
                
        for i in range(N-2,-1,-1):
            for j in range(N):
                bottom_counts[(i,j)] = (bottom_counts[(i+1,j)]+grid[(i+1,j)]) * int(grid[(i,j)] == 1)

        for i in range(N):
            for j in range(1,N):
                left_counts[(i,j)] = (left_counts[(i,j-1)]+grid[(i,j-1)]) * int(grid[(i,j)] == 1)

        for i in range(N):
            for j in range(N-2,-1,-1):
                right_counts[(i,j)] = (right_counts[(i,j+1)]+grid[(i,j+1)]) * int(grid[(i,j)] == 1)
                
        max_plus = 0
        for i in range(N):
            for j in range(N):
                max_plus = max(max_plus,min(top_counts[(i,j)],bottom_counts[(i,j)],left_counts[(i,j)],right_counts[(i,j)]))
                   
        return max_plus+1
```

## 766. Toeplitz Matrix
A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same element.

Now given an M x N matrix, return True if and only if the matrix is Toeplitz.
>Example 1:
```
Input: matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]]
Output: True
Explanation:
1234
5123
9512
```

In the above grid, the diagonals are "[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]", and in each diagonal all elements are the same, so the answer is True.

>Example 2:
```
Input: matrix = [[1,2],[2,2]]
Output: False
```

Explanation:
The diagonal "[1, 2]" has different elements.

Note:

- matrix will be a 2D array of integers.
- matrix will have a number of rows and columns in range [1, 20].
- matrix[i][j] will be integers in range [0, 99].


```python
class Solution(object):
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        L = len(matrix)
        for i in range(1,L):
            if matrix[i-1][:-1] != matrix[i][1:]:
                return False
        return True
```


## 771. Jewels and Stones
You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".
>
Example 1:
```
Input: J = "aA", S = "aAAbbbb"
Output: 3
```
>
Example 2:
```
Input: J = "z", S = "ZZ"
Output: 0
```

Note:

- S and J will consist of letters and have length at most 50.
- The characters in J are distinct.

```python
class Solution(object):
    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        from collections import defaultdict
        jewels = defaultdict(int)
        for j in J:
            jewels[j] += 1
            
        count = 0
        for s in S:
            if s in jewels:
                count += 1
                
        return count
```

## 779. K-th Symbol in Grammar
On the first row, we write a 0. Now in every subsequent row, we look at the previous row and replace each occurrence of 0 with 01, and each occurrence of 1 with 10.

Given row N and index K, return the K-th indexed symbol in row N. (The values of K are 1-indexed.) (1 indexed).

>
Examples:
```
Input: N = 1, K = 1
Output: 0
Input: N = 2, K = 1
Output: 0
Input: N = 2, K = 2
Output: 1
Input: N = 4, K = 5
Output: 1
Explanation:
row 1: 0
row 2: 01
row 3: 0110
row 4: 01101001
```
Note:
1. N will be an integer in the range [1, 30].
2. K will be an integer in the range [1, 2^(N-1)].

```python
class Solution(object):
    def kthGrammar(self, N, K):
        """
        :type N: int
        :type K: int
        :rtype: int
        """
        # Note than Nth row has 2*N indices. If K <= N/2, symbol(N,K) = symbol(N-1,K), else  = 1-symbol(N-1,K-N/2)
        multiplier = 0
        for n in range(N-1,0,-1):
            if K > 2**(n-1):
                multiplier = not multiplier
                K -= 2**(n-1)

        return int(multiplier)
```

## 781. Rabbits in Forest
In a forest, each rabbit has some color. Some subset of rabbits (possibly all of them) tell you how many other rabbits have the same color as them. Those answers are placed in an array.

Return the minimum number of rabbits that could be in the forest.

>
Examples:
```
Input: answers = [1, 1, 2]
Output: 5
```
Explanation:
The two rabbits that answered "1" could both be the same color, say red.
The rabbit than answered "2" can't be red or the answers would be inconsistent.
Say the rabbit that answered "2" was blue.
Then there should be 2 other blue rabbits in the forest that didn't answer into the array.
The smallest possible number of rabbits in the forest is therefore 5: 3 that answered plus 2 that didn't.
```
Input: answers = [10, 10, 10]
Output: 11
```
```
Input: answers = []
Output: 0
```

Note:
- Answers will have length at most 1000.
- Each answers[i] will be an integer in the range [0, 999].

```python
class Solution(object):
    def numRabbits(self, answers):
        """
        :type answers: List[int]
        :rtype: int
        """
        # A mathematical solution
        from collections import defaultdict
        counts = defaultdict(int)
        for a in answers:
            counts[a] += 1

        sum = 0
        for k in counts.keys():
            sum += math.ceil(counts[k]/(k+1.))*(k+1)
        return int(sum)
```

## 783. Minimum Distance Between BST Nodes
Given a Binary Search Tree (BST) with the root node root, return the minimum difference between the values of any two different nodes in the tree.

>Example :
```
Input: root = [4,2,6,1,3,null,null]
Output: 1
Explanation:
Note that root is a TreeNode object, not an array.
The given tree [4,2,6,1,3,null,null] is represented by the following diagram:
          4
        /   \
      2      6
     / \    
    1   3  
while the minimum difference in this tree is 1, it occurs between node 1 and node 2, also between node 3 and node 2.
```

Note:
- The size of the BST will be between 2 and 100.
- The BST is always valid, each node's value is an integer, and each node's value is different.

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDiffInBST(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        current_val = -float('inf')
        mindiff = float('inf')
        return self.minDiff(root,current_val,mindiff)[1]

    def minDiff(self,node,val,diff):
        if node == None:
            return val,diff

        val,diff = self.minDiff(node.left,val,diff)
        diff = min(diff,node.val-val)
        val = node.val
        val,diff = self.minDiff(node.right,val,diff)

        return val,diff
```

## 785. Is Graph Bipartite?
Given an undirected graph, return true if and only if it is bipartite.

Recall that a graph is bipartite if we can split it's set of nodes into two independent subsets A and B such that every edge in the graph has one node in A and another node in B.

The graph is given in the following form: graph[i] is a list of indexes j for which the edge between nodes i and j exists.  Each node is an integer between 0 and graph.length - 1.  There are no self edges or parallel edges: graph[i] does not contain i, and it doesn't contain any element twice.
>
Example 1:
```
Input: [[1,3], [0,2], [1,3], [0,2]]
Output: true
Explanation: 
The graph looks like this:
0----1
|    |
|    |
3----2
We can divide the vertices into two groups: {0, 2} and {1, 3}.
```
>
Example 2:
```
Input: [[1,2,3], [0,2], [0,1,3], [0,2]]
Output: false
Explanation: 
The graph looks like this:
0----1
| \  |
|  \ |
3----2
We cannot find a way to divide the set of nodes into two independent subsets.
``` 

Note:

- graph will have length in range [1, 100].
- graph[i] will contain integers in range [0, graph.length - 1].
- graph[i] will not contain i or duplicate values.
- The graph is undirected: if any element j is in graph[i], then i will be in graph[j].

```python
class Solution(object):
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """
        # Bipartite graph can be covered with two colors.
        valid_nodes = [idx for idx in range(len(graph)) if graph[idx] != []]
        if valid_nodes == []: # empty graph
            return True
        
        visited = {}
        colors = {}
        
        current_nodes = [valid_nodes[0]]
        visited[valid_nodes[0]] = 1
        
        while current_nodes != []:
            colors[current_nodes[0]] = 0

            for node in current_nodes:
                current_color = colors[node]
                for next_node in graph[node]:
                    if next_node not in visited:
                        visited[next_node] = 1

                        if next_node not in colors:
                            colors[next_node] = 1-current_color

                        current_nodes += [next_node]
                    else:
                        if colors[next_node] != 1 - current_color:
                            return False

            # For nodes not visited during the BFS phase
            current_nodes = [g for g in valid_nodes if g not in visited]
            if current_nodes != []:
                current_nodes = [current_nodes[0]]
                
        return True
```

## 789. Escape The Ghosts
You are playing a simplified Pacman game. You start at the point (0, 0), and your destination is (target[0], target[1]). There are several ghosts on the map, the i-th ghost starts at (ghosts[i][0], ghosts[i][1]).

Each turn, you and all ghosts simultaneously *may* move in one of 4 cardinal directions: north, east, west, or south, going from the previous point to a new point 1 unit of distance away.

You escape if and only if you can reach the target before any ghost reaches you (for any given moves the ghosts may take.)  If you reach any square (including the target) at the same time as a ghost, it doesn't count as an escape.

Return True if and only if it is possible to escape.
>
Example 1:
```
Input:
ghosts = [[1, 0], [0, 3]]
target = [0, 1]
Output: true
```
Explanation:
You can directly reach the destination (0, 1) at time 1, while the ghosts located at (1, 0) or (0, 3) have no way to catch up with you.
>
Example 2:
```
Input:
ghosts = [[1, 0]]
target = [2, 0]
Output: false
```
Explanation:
You need to reach the destination (2, 0), but the ghost at (1, 0) lies between you and the destination.
>
Example 3:
```
Input:
ghosts = [[2, 0]]
target = [1, 0]
Output: false
```
Explanation:
The ghost can reach the target at the same time as you.

Note:
- All points have coordinates with absolute value <= 10000.
- The number of ghosts will not exceed 100.

```python
class Solution(object):
    def escapeGhosts(self, ghosts, target):
        """
        :type ghosts: List[List[int]]
        :type target: List[int]
        :rtype: bool
        """
        # If path from ghose to destination os shorter, there will be no escape since the ghost can simply reach the destination
        # and wait to capture.
        # Also, time from (x,y) to (u,v) is simply |u-x| + |v-y|
        myTime = abs(target[0]) + abs(target[1])
        ghost_minTime = min([abs(g[0] - target[0]) + abs(g[1] - target[1]) for g in ghosts])

        return myTime < ghost_minTime
```

## 794. Valid Tic-Tac-Toe State

A Tic-Tac-Toe board is given as a string array board. Return True if and only if it is possible to reach this board position during the course of a valid tic-tac-toe game.

The board is a 3 x 3 array, and consists of characters " ", "X", and "O".  The " " character represents an empty square.

Here are the rules of Tic-Tac-Toe:

Players take turns placing characters into empty squares (" ").
The first player always places "X" characters, while the second player always places "O" characters.
"X" and "O" characters are always placed into empty squares, never filled ones.
The game ends when there are 3 of the same (non-empty) character filling any row, column, or diagonal.
The game also ends if all squares are non-empty.
No more moves can be played if the game is over.
>
Example 1:
```
Input: board = ["O  ", "   ", "   "]
Output: false
Explanation: The first player always plays "X".
```
>
Example 2:
```
Input: board = ["XOX", " X ", "   "]
Output: false
Explanation: Players take turns making moves.
```
>
Example 3:
```
Input: board = ["XXX", "   ", "OOO"]
Output: false
```
>
Example 4:
```
Input: board = ["XOX", "O O", "XOX"]
Output: true
```

Note:
- board is a length-3 array of strings, where each string board[i] has length 3.
- Each board[i][j] is a character in the set {" ", "X", "O"}.

```python
class Solution(object):
    def validTicTacToe(self, board):
        """
        :type board: List[str]
        :rtype: bool
        """
        numX = 0
        numO = 0
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'X':
                    numX += 1
                elif board[i][j] == 'O':
                    numO += 1

        if numX > numO + 1 or numX < numO:
            return False

        X_matches = 0
        O_matches = 0

        def check_match(board,a,b,c):
            X_matches = 0
            O_matches = 0
            if board[a[0]][a[1]]==board[b[0]][b[1]]==board[c[0]][c[1]]:
                if board[a[0]][a[1]] == 'X':
                    X_matches = 1
                elif board[a[0]][a[1]] == 'O':
                    O_matches = 1
            else:
                return 0,0
            return X_matches, O_matches

        X_matches += check_match(board,(0,0),(0,1),(0,2))[0]
        O_matches += check_match(board,(0,0),(0,1),(0,2))[1]

        X_matches += check_match(board,(1,0),(1,1),(1,2))[0]
        O_matches += check_match(board,(1,0),(1,1),(1,2))[1]

        X_matches += check_match(board,(2,0),(2,1),(2,2))[0]
        O_matches += check_match(board,(2,0),(2,1),(2,2))[1]

        X_matches += check_match(board,(0,0),(1,0),(2,0))[0]
        O_matches += check_match(board,(0,0),(1,0),(2,0))[1]

        X_matches += check_match(board,(0,1),(1,1),(2,1))[0]
        O_matches += check_match(board,(0,1),(1,1),(2,1))[1]

        X_matches += check_match(board,(0,2),(1,2),(2,2))[0]
        O_matches += check_match(board,(0,2),(1,2),(2,2))[1]

        X_matches += check_match(board,(0,0),(1,1),(2,2))[0]
        O_matches += check_match(board,(0,0),(1,1),(2,2))[1]

        X_matches += check_match(board,(0,2),(1,1),(2,0))[0]
        O_matches += check_match(board,(0,2),(1,1),(2,0))[1]

        if X_matches > 0 and O_matches > 0:
            return False

        if X_matches > 0 and numX == numO:
            return False

        if O_matches > 0 and numX != numO:
            return False


        return True
```

## 813. Largest Sum of Averages
We partition a row of numbers A into at most K adjacent (non-empty) groups, then our score is the sum of the average of each group. What is the largest score we can achieve?

Note that our partition must use every number in A, and that scores are not necessarily integers.

>
Example
```
Input: 
A = [9,1,2,3,9]
K = 3
Output: 20
Explanation: 
The best choice is to partition A into [9], [1, 2, 3], [9]. The answer is 9 + (1 + 2 + 3) / 3 + 9 = 20.
We could have also partitioned A into [9, 1], [2], [3, 9], for example.
That partition would lead to a score of 5 + 2 + 6 = 13, which is worse.
```
Note:

- 1 <= A.length <= 100.
- 1 <= A[i] <= 10000.
- 1 <= K <= A.length.
- Answers within 10^-6 of the correct answer will be accepted as correct.

```python
class Solution(object):
    def largestSumOfAverages(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: float
        """
        # Dynamic programming
        N = len(A)
        largest_scores = [[0 for _ in range(N)] for _ in range(K)]
        largest_scores[0][0] = A[0]
        for j in range(1,N):
            largest_scores[0][j] = (A[j] + largest_scores[0][j-1]*j)/(j+1.0)
        
        for i in range(1,K):
            for j in range(N-1,i-1,-1):
                running_sum = 0
                counter = 0
                for k in range(j-1,i-2,-1):
                    running_sum += A[k+1]
                    counter +=1.0
                    running_mean = running_sum/counter
                    largest_scores[i][j] = max(largest_scores[i][j], running_mean+largest_scores[i-1][k])
        return largest_scores[-1][-1]
```

## 814. Binary Tree Pruning

We are given the head node root of a binary tree, where additionally every node's value is either a 0 or a 1.

Return the same tree where every subtree (of the given tree) not containing a 1 has been removed.

(Recall that the subtree of a node X is X, plus every node that is a descendant of X.)

>
Example 1:
```
Input: [1,null,0,0,1]
Output: [1,null,0,null,1]
Explanation: 
Only the red nodes satisfy the property "every subtree not containing a 1".
The diagram on the right represents the answer.
```
![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/04/06/1028_2.png)

>
Example 2:
```
Input: [1,0,1,0,0,0,1]
Output: [1,null,1,null,1]
```
![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/04/06/1028_1.png)

>
Example 3:
```
Input: [1,1,0,1,1,0,1,0]
Output: [1,1,0,1,1,null,1]
```
![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/04/05/1028.png)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pruneTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        # Perform post-order traversal and remove nodes
        root = self.post_order(root)
        if root.val == 0 and root.left == None and root.right == None:
            return None
        else:
            return root
        
    def post_order(self,node):
        if node == None:
            return node
        self.post_order(node.left)
        self.post_order(node.right)
        self.process(node)
        return node
        
    def process(self,node):
        if node.left != None:
            if node.left.val == -1:
                node.left = None
        if node.right != None:
            if node.right.val == -1:
                node.right = None
        if node.left == None and node.right == None:
            if node.val == 0:
                node.val = -1
```

## 826. Most Profit Assigning Work
We have jobs: difficulty[i] is the difficulty of the ith job, and profit[i] is the profit of the ith job. 

Now we have some workers. worker[i] is the ability of the ith worker, which means that this worker can only complete a job with difficulty at most worker[i]. 

Every worker can be assigned at most one job, but one job can be completed multiple times.

For example, if 3 people attempt the same job that pays $1, then the total profit will be $3.  If a worker cannot complete any job, his profit is $0.

What is the most profit we can make?
>
Example 1:
```
Input: difficulty = [2,4,6,8,10], profit = [10,20,30,40,50], worker = [4,5,6,7]
Output: 100 
Explanation: Workers are assigned jobs of difficulty [4,4,6,6] and they get profit of [20,20,30,30] seperately.
```

Notes:

- 1 <= difficulty.length = profit.length <= 10000
- 1 <= worker.length <= 10000
- difficulty[i], profit[i], worker[i]  are in range [1, 10^5]

```python
class Solution(object):
    def maxProfitAssignment(self, difficulty, profit, worker):
        """
        :type difficulty: List[int]
        :type profit: List[int]
        :type worker: List[int]
        :rtype: int
        """
        # Since the same job can be completed multiple times, we can simply use a greedy algorithm
        # For each worker, find the job that gives maximum profit amongst lower difficulties
        maxProfit = 0
        difficulty_profit = sorted(zip(difficulty,profit),key=lambda x:x[0])
        worker = sorted(worker)
        P = len(difficulty_profit)
        W = len(worker)
        current_max_profit = 0

        idx = 0
        
        for i in range(W):
            while idx < P and worker[i] >= difficulty_profit[idx][0]:
                current_max_profit = max(current_max_profit,difficulty_profit[idx][1])
                idx += 1
            maxProfit += current_max_profit

        return maxProfit
```

## 829. Consecutive Numbers Sum
Given a positive integer N, how many ways can we write it as a sum of consecutive positive integers?
>
Example 1:
```
Input: 5
Output: 2
Explanation: 5 = 5 = 2 + 3
```
>
Example 2:
```
Input: 9
Output: 3
Explanation: 9 = 9 = 4 + 5 = 2 + 3 + 4
```
>
Example 3:
```
Input: 15
Output: 4
Explanation: 15 = 15 = 8 + 7 = 4 + 5 + 6 = 1 + 2 + 3 + 4 + 5
```

Note: 1 <= N <= 10 ^ 9.

```python
class Solution(object):
    def consecutiveNumbersSum(self, N):
        """
        :type N: int
        :rtype: int
        """
        # There is always 1 way (=N).
        # If N = x+(x+1), there's another way, i.e N % 2 == 1
        # If N = (x-1)+x+(x+1), there's a way, i.e., N % 3 == 0
        # Continuing..
        # If k is odd, N % k == 0 gives one way
        # If k is even, N % k == k/2 gives one way
        # We only need to test for k = 2,...,j s.t 1+2+...+j=N
        # k can be upper bounded by 2*sqrt(N)
        
        num_ways = 1

        for k in range(2,int(2*math.sqrt(N))):
            if k**2+k > 2*N:
                return num_ways
            if N % k == 0 and k % 2 == 1:
                num_ways += 1
            elif N % k == k/2 and k % 2 == 0:
                num_ways += 1
                
        return num_ways
```

## 841. Keys and Rooms
There are N rooms and you start in room 0.  Each room has a distinct number in 0, 1, 2, ..., N-1, and each room may have some keys to access the next room. 

Formally, each room i has a list of keys rooms[i], and each key rooms[i][j] is an integer in [0, 1, ..., N-1] where N = rooms.length.  A key rooms[i][j] = v opens the room with number v.

Initially, all the rooms start locked (except for room 0). 

You can walk back and forth between rooms freely.

Return true if and only if you can enter every room.
>
Example 1:
```
Input: [[1],[2],[3],[]]
Output: true
Explanation:  
We start in room 0, and pick up key 1.
We then go to room 1, and pick up key 2.
We then go to room 2, and pick up key 3.
We then go to room 3.  Since we were able to go to every room, we return true.
```

>
Example 2:
```
Input: [[1,3],[3,0,1],[2],[0]]
Output: false
Explanation: We can't enter the room with number 2.
```

Note:

- 1 <= rooms.length <= 1000
- 0 <= rooms[i].length <= 1000
- The number of keys in all rooms combined is at most 3000.

```python
class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        if rooms == [[]]:
            return True
        # Graph traversal problem using DFS. BFS will be easier, hence trying DFS
        visited = {0: 1} # dictionary of visited rooms
        visited = self.visit_rooms(0, rooms, visited)
        print visited
        if len(visited) == len(rooms):
            return True
        return False
        
    def visit_rooms(self, current_room, rooms, visited):
        if rooms[current_room] is None:
            return visited
        for room in rooms[current_room]:
            if room not in visited:
                visited[room] = 1
                visited = self.visit_rooms(room, rooms, visited)
            
        return visited
```

## 845. Longest Mountain in Array
Let's call any (contiguous) subarray B (of A) a mountain if the following properties hold:

- B.length >= 3
- There exists some 0 < i < B.length - 1 such that B[0] < B[1] < ... B[i-1] < B[i] > B[i+1] > ... > B[B.length - 1]
(Note that B could be any subarray of A, including the entire array A.)

Given an array A of integers, return the length of the longest mountain. 

Return 0 if there is no mountain.

>
Example 1:
```
Input: [2,1,4,7,3,2,5]
Output: 5
Explanation: The largest mountain is [1,4,7,3,2] which has length 5.
```

>
Example 2:
```
Input: [2,2,2]
Output: 0
Explanation: There is no mountain.
```

Note:

1. 0 <= A.length <= 10000
2. 0 <= A[i] <= 10000

```python
class Solution(object):
    def longestMountain(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        N = len(A)
        if N < 3:
            return 0
        left_slopes = [0] * N
        right_slopes = [0]*N
        
        for i in range(1,N):
            if A[i] > A[i-1]:
                left_slopes[i] = left_slopes[i-1] + 1
            else:
                left_slopes[i] = 0
        
        for i in range(N-2,0,-1):
            if A[i] > A[i+1]:
                right_slopes[i] = right_slopes[i+1] + 1
            else:
                right_slopes[i] = 0
                
        return max(map(lambda (x,y): (x+y+1)*(min(x,y)>0),zip(left_slopes, right_slopes)))
```

## 848. Shifting Letters
We have a string S of lowercase letters, and an integer array shifts.
Call the shift of a letter, the next letter in the alphabet, (wrapping around so that 'z' becomes 'a'). 
For example, shift('a') = 'b', shift('t') = 'u', and shift('z') = 'a'.
Now for each shifts[i] = x, we want to shift the first i+1 letters of S, x times.
Return the final string after all such shifts to S are applied.

>Example 1:
```
Input: S = "abc", shifts = [3,5,9]
Output: "rpl"
Explanation: 
We start with "abc".
After shifting the first 1 letters of S by 3, we have "dbc".
After shifting the first 2 letters of S by 5, we have "igc".
After shifting the first 3 letters of S by 9, we have "rpl", the answer.
```

Note:
- 1 <= S.length = shifts.length <= 20000
- 0 <= shifts[i] <= 10 ^ 9

```python
class Solution(object):
    def shiftingLetters(self, S, shifts):
        """
        :type S: str
        :type shifts: List[int]
        :rtype: str
        """
        solution = ''
        net_shift = sum(shifts)
        solution += self.shift(S[0], net_shift)
        for i in range(1,len(shifts)):
            net_shift -= shifts[i-1]
            solution += self.shift(S[i], net_shift)
        return solution
    
    def shift(self, s, n):
        return chr(ord(s)+n%26+ord('a')-ord('z')-1) if ord(s)+n%26 > ord('z') else chr(ord(s)+n%26)
```

## 849. Maximize Distance to Closest Person
In a row of seats, 1 represents a person sitting in that seat, and 0 represents that the seat is empty. 
There is at least one empty seat, and at least one person sitting.
Alex wants to sit in the seat such that the distance between him and the closest person to him is maximized. 
Return that maximum distance to closest person.

>Example 1:
```
Input: [1,0,0,0,1,0,1]
Output: 2
Explanation: 
If Alex sits in the second open seat (seats[2]), then the closest person has distance 2.
If Alex sits in any other open seat, the closest person has distance 1.
Thus, the maximum distance to the closest person is 2.
```
>Example 2:
```
Input: [1,0,0,0]
Output: 3
Explanation: 
If Alex sits in the last seat, the closest person is 3 seats away.
This is the maximum distance possible, so the answer is 3.
```

Note:
- 1 <= seats.length <= 20000
- seats contains only 0s or 1s, at least one 0, and at least one 1.

```python
class Solution(object):
    def maxDistToClosest(self, seats):
        """
        :type seats: List[int]
        :rtype: int
        """
        start = 0
        max_dist = 0
        init = 0
        init_flag = False
        
        for i in range(len(seats)):
            if seats[i] == 1:
                max_dist = max(max_dist,i-start)
                if not init_flag:
                    init = i-start
                    init_flag = True
                start = i
        final = i-start
        print init, max_dist, final
        return max(init, max_dist/2, final)
```

## 853. Car Fleet
N cars are going to the same destination along a one lane road.  The destination is target miles away.

Each car i has a constant speed speed[i] (in miles per hour), and initial position position[i] miles towards the target along the road.

A car can never pass another car ahead of it, but it can catch up to it, and drive bumper to bumper at the same speed.

The distance between these two cars is ignored - they are assumed to have the same position.

A car fleet is some non-empty set of cars driving at the same position and same speed.  Note that a single car is also a car fleet.

If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.

How many car fleets will arrive at the destination?
>
Example 1:
```
Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
Output: 3
Explanation:
The cars starting at 10 and 8 become a fleet, meeting each other at 12.
The car starting at 0 doesn't catch up to any other car, so it is a fleet by itself.
The cars starting at 5 and 3 become a fleet, meeting each other at 6.
Note that no other cars meet these fleets before the destination, so the answer is 3.
```

Note:

- 0 <= N <= 10 ^ 4
- 0 < target <= 10 ^ 6
- 0 < speed[i] <= 10 ^ 6
- 0 <= position[i] < target
- All initial positions are different.

```python
class Solution(object):
    def carFleet(self, target, position, speed):
        """
        :type target: int
        :type position: List[int]
        :type speed: List[int]
        :rtype: int
        """
        N = len(speed)
        if N < 2:
            return N
        # Calculate time to destination in no traffic
        time_to_dest = []
        for i in range(N):
            time_to_dest.append((target-position[i])/(speed[i]+0.0))
            
        # Sort by position and count the number of fleets
        positions_and_times = sorted(zip(position,time_to_dest), key = lambda x: x[0])
        
        num_fleets = 1
        current_time = positions_and_times[-1][1]
        for i in range(N-2,-1,-1):
            if positions_and_times[i][1] > current_time:
                num_fleets += 1
                current_time = positions_and_times[i][1]
                
        return num_fleets
```

## 855. Exam Room
In an exam room, there are N seats in a single row, numbered 0, 1, 2, ..., N-1.

When a student enters the room, they must sit in the seat that maximizes the distance to the closest person.  If there are multiple such seats, they sit in the seat with the lowest number.  (Also, if no one is in the room, then the student sits at seat number 0.)

Return a class ExamRoom(int N) that exposes two functions: ExamRoom.seat() returning an int representing what seat the student sat in, and ExamRoom.leave(int p) representing that the student in seat number p now leaves the room.  It is guaranteed that any calls to ExamRoom.leave(p) have a student sitting in seat p.

>
Example 1:
```
Input: ["ExamRoom","seat","seat","seat","seat","leave","seat"], [[10],[],[],[],[],[4],[]]
Output: [null,0,9,4,2,null,5]
Explanation:
ExamRoom(10) -> null
seat() -> 0, no one is in the room, then the student sits at seat number 0.
seat() -> 9, the student sits at the last seat number 9.
seat() -> 4, the student sits at the last seat number 4.
seat() -> 2, the student sits at the last seat number 2.
leave(4) -> null
seat() -> 5, the student sits at the last seat number 5.
```

Note:
- 1 <= N <= 10^9
- ExamRoom.seat() and ExamRoom.leave() will be called at most 10^4 times across all test cases.
- Calls to ExamRoom.leave(p) are guaranteed to have a student currently sitting in seat number p.

```python
class ExamRoom(object):

    def __init__(self, N):
        """
        :type N: int
        """
        self.num_students = -1
        self.N = N
        self.locations = []        

    def seat(self):
        """
        :rtype: int
        """
        self.num_students += 1
        if self.num_students == 0:
            self.locations = [0]
            return 0
        else:
            L = len(self.locations)
            max_dist = 0

            if self.locations[0] > 0:
                new_location = 0
                max_dist = self.locations[0]
                                
            for idx in range(1,L):
                if self.locations[idx] - self.locations[idx-1] == 1:
                    continue
                else:
                    location = (self.locations[idx]+self.locations[idx-1])/2
                    dist = math.ceil((self.locations[idx]-self.locations[idx-1])/2)
                    if dist > max_dist:
                        max_dist = dist
                        new_location = location
                    
            if self.locations[-1] < self.N-1:
                location = self.N-1
                if location - self.locations[-1] > max_dist:
                    new_location = location
                        
            # insert new location in locations
            if new_location < self.locations[0]:
                self.locations = [new_location]+self.locations
                return new_location
            elif new_location > self.locations[-1]:
                self.locations = self.locations+[new_location]
                return new_location
            for idx in range(1,L):
                if self.locations[idx]>new_location and self.locations[idx-1]<new_location:
                    self.locations = self.locations[:idx]+[new_location]+self.locations[idx:]
                    return new_location

    def leave(self, p):
        """
        :type p: int
        :rtype: void
        """
        self.num_students -= 1
        self.locations.remove(p)

# Your ExamRoom object will be instantiated and called as such:
# obj = ExamRoom(N)
# param_1 = obj.seat()
# obj.leave(p)
```

## 860. Lemonade Change
At a lemonade stand, each lemonade costs $5. 

Customers are standing in a queue to buy from you, and order one at a time (in the order specified by bills).

Each customer will only buy one lemonade and pay with either a $5, $10, or $20 bill.  You must provide the correct change to each customer, so that the net transaction is that the customer pays $5.

Note that you don't have any change in hand at first.

Return true if and only if you can provide every customer with correct change.

>
Example 1:
```
Input: [5,5,5,10,20]
Output: true
Explanation: 
From the first 3 customers, we collect three $5 bills in order.
From the fourth customer, we collect a $10 bill and give back a $5.
From the fifth customer, we give a $10 bill and a $5 bill.
Since all customers got correct change, we output true.
```
>
Example 2:
```
Input: [5,5,10]
Output: true
```
>
Example 3:
```
Input: [10,10]
Output: false
```
>
Example 4:
```
Input: [5,5,10,10,20]
Output: false
Explanation: 
From the first two customers in order, we collect two $5 bills.
For the next two customers in order, we collect a $10 bill and give back a $5 bill.
For the last customer, we can't give change of $15 back because we only have two $10 bills.
Since not every customer received correct change, the answer is false.
```

Note:

- 0 <= bills.length <= 10000
- bills[i] will be either 5, 10, or 20.

```python
class Solution(object):
    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
        L = len(bills)
        from collections import defaultdict
        notes = defaultdict(int)
        for i in range(L):
            notes[bills[i]] += 1
            change = bills[i] - 5
            if change == 0:
                continue
            elif change == 5:
                if notes[5] == 0:
                    return False
                else:
                    notes[5] -= 1
            elif change == 15:
                if notes[10] > 0 and notes[5] > 0:
                    notes[10] -= 1
                    notes[5] -= 1
                elif notes[5] >= 3:
                    notes[5] -= 3
                else:
                    return False
        return True
```

## 867. Transpose Matrix
Given a matrix A, return the transpose of A.

The transpose of a matrix is the matrix flipped over it's main diagonal, switching the row and column indices of the matrix.
>
Example 1:
```
Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: [[1,4,7],[2,5,8],[3,6,9]]
```
>
Example 2:
```
Input: [[1,2,3],[4,5,6]]
Output: [[1,4],[2,5],[3,6]]
```

Note:

- 1 <= A.length <= 1000
- 1 <= A[0].length <= 1000

```python
class Solution(object):
    def transpose(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        M = len(A)
        N = len(A[0])
        
        solution = []
        for i in range(N):
            solution += [[A[j][i] for j in range(M)]]
            
        return solution
```

## 869. Reordered Power of 2
Starting with a positive integer N, we reorder the digits in any order (including the original order) such that the leading digit is not zero.

Return true if and only if we can do this in a way such that the resulting number is a power of 2.
>
Example 1:
```
Input: 1
Output: true
```
>
Example 2:
```
Input: 10
Output: false
```
>
Example 3:
```
Input: 16
Output: true
```
>
Example 4:
```
Input: 24
Output: false
```
>
Example 5:
```
Input: 46
Output: true
```

Note:

- 1 <= N <= 10^9

```python
class Solution(object):
    def reorderedPowerOf2(self, N):
        """
        :type N: int
        :rtype: bool
        """
        from collections import Counter
        
        # N < 10e9, i.e., N < 2^30. So, we can simply check if N is in that list.
        hard_coded_list = [2**x for x in range(30)]
        
        c = Counter(str(N))
        for s in hard_coded_list:
            if c == Counter(str(s)):
                return True
        return False
```

## 872. Leaf-Similar Trees
Consider all the leaves of a binary tree.  From left to right order, the values of those leaves form a leaf value sequence.

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/16/tree.png)

For example, in the given tree above, the leaf value sequence is (6, 7, 4, 9, 8).

Two binary trees are considered leaf-similar if their leaf value sequence is the same.

Return true if and only if the two given trees with head nodes root1 and root2 are leaf-similar.

Note:

- Both of the given trees will have between 1 and 100 nodes.

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def leafSimilar(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: bool
        """
        return self.get_leaf_sequence(root1,[]) == self.get_leaf_sequence(root2,[])
        
    def get_leaf_sequence(self,root,seq):
        if root == None:
            return []
        self.get_leaf_sequence(root.left,seq)
        if root.left == None and root.right == None:
            seq += [root.val]
        self.get_leaf_sequence(root.right,seq)
        
        return seq
```

## 873. Length of Longest Fibonacci Subsequence
A sequence X_1, X_2, ..., X_n is fibonacci-like if:

n >= 3
X_i + X_{i+1} = X_{i+2} for all i + 2 <= n
Given a strictly increasing array A of positive integers forming a sequence, find the length of the longest fibonacci-like subsequence of A.  If one does not exist, return 0.

(Recall that a subsequence is derived from another sequence A by deleting any number of elements (including none) from A, without changing the order of the remaining elements.  For example, [3, 5, 8] is a subsequence of [3, 4, 5, 6, 7, 8].)

>
Example 1:
```
Input: [1,2,3,4,5,6,7,8]
Output: 5
Explanation:
The longest subsequence that is fibonacci-like: [1,2,3,5,8].
```
>
Example 2:
```
Input: [1,3,7,11,12,14,18]
Output: 3
Explanation:
The longest subsequence that is fibonacci-like:
[1,11,12], [3,11,14] or [7,11,18].
```

Note:
- 3 <= A.length <= 1000
- 1 <= A[0] < A[1] < ... < A[A.length - 1] <= 10^9

```python

```

## 875. Koko Eating Bananas
Koko loves to eat bananas.  There are N piles of bananas, the i-th pile has piles[i] bananas.  The guards have gone and will come back in H hours.

Koko can decide her bananas-per-hour eating speed of K.  Each hour, she chooses some pile of bananas, and eats K bananas from that pile.  If the pile has less than K bananas, she eats all of them instead, and won't eat any more bananas during this hour.

Koko likes to eat slowly, but still wants to finish eating all the bananas before the guards come back.

Return the minimum integer K such that she can eat all the bananas within H hours.

>
Example 1:
```
Input: piles = [3,6,7,11], H = 8
Output: 4
```
>
Example 2:
```
Input: piles = [30,11,23,4,20], H = 5
Output: 30
```
>
Example 3:
```
Input: piles = [30,11,23,4,20], H = 6
Output: 23
```

Note:
- 1 <= piles.length <= 10^4
- piles.length <= H <= 10^9
- 1 <= piles[i] <= 10^9

```python
class Solution(object):
    def minEatingSpeed(self, piles, H):
        """
        :type piles: List[int]
        :type H: int
        :rtype: int
        """
        # piles: x1, x2, ...., xN
        # if K bananas are eaten evey time, sum_i(ceil(x_i/K)) <= H. Need to minimize K.
        # 1 <= K <= max(x_i). Try for each value of K.
        # Even better, use binry search for searching for best K.
        
        high = max(piles)
        low = 1
    
        if self.f(1,piles) <= H:
            return 1
        return self.binary_search(piles,H,low,high)

    def f(self,k,piles):
        return sum([(p-1)/k + 1 for p in piles])
                    
    def binary_search(self,piles,H,low,high):
        mid = (low+high)/2
        if self.f(low,piles) == H:
            return low
        elif high == low + 1:
            return high
        elif self.f(mid,piles) <= H:
            return self.binary_search(piles,H,low,mid)
        elif self.f(mid,piles) > H:
            low = mid
            return self.binary_search(piles,H,mid,high)
```
