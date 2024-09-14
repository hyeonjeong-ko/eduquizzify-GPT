import sys

N = int(input())
lst = list(map(int, input().split()))

if max(lst) < 0: # 모두 음수인 경우
    print(max(lst)) # 1개만 더한 값이 max
else:
    dp = [-1000] * N

    dp[0] = lst[0]

    for i in range(1, N):
        # 최대연속합에 현재수 더하는 경우 / 새로운 연속합 시작하는 경우
        dp[i] = max(dp[i - 1] + lst[i], lst[i])

    print(max(dp))
