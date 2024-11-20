import sys

print("Python", "Java", file=sys.stdout)
print("Python", "Java", file=sys.stderr)


scores = {"수학":0, "영어":50, "코딩":70}
for subject, score in scores.items():
    print(subject.ljust(8), str(score).rjust(4), sep=":")

# 은행 대기순번표
# 001, 002, 003
for num in range(1, 21):
    print("대기번호 : ", str(num).zfill(3))

# answer = input("아무값이나 입력하세요. : ")
# print("입력하신 값은 : {}".format(answer))

# 빈 자리는 빈공간으로 두고, 오른쪽 정렬을 하되, 총 10자리 공간을 확보
space = "asd"
print(space.rjust(10))
print("{0: >10}".format(500))
# 양수일 땐 +로 표시, 음수일 땐 -로 표시
print({"{0: >+10}".format(500)})
print({"{0: >-10}".format(50 - 100000)})
# 왼쪽 정렬하고, 빈칸으로 _로 채움
print("{0:_<10}".format(500))
#3자리마다 , 찍어주기
print("{0:,}".format(1000000))
print("{0:+,}".format(-1000000))
#3자리 마다 콤마를 찍어주기, 부호도 붙이고, 자릿수 확보하기
#빈자리는 ^로 표시
print("{0:^<30,}".format(10000000000000))
# 소수점 출력
print("{0:f}".format(5/3))
# 소수점을 특정 자리수까지 표시
print("{0:.2f}".format(5/3))

def man_unit(num):
    num = str(num)[::-1]
    list = [num[i:i+4] for i in range(0, len(num), 4)]
    print(list)
    num = ",".join(list)[::-1]
    print(num)

man_unit(100000000)