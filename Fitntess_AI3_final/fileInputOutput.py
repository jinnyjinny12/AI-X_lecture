# w는 쓰기 이미 내용이 있다면 덮어 쓰기
score_file = open("score.txt", "w", encoding="utf-8")
print("수학 : 0", file=score_file)
print("영어 : 50", file=score_file)
score_file.close()
# a 이미 쓴 파일에 추가하기
score_file = open("score.txt", "a", encoding="utf-8")
print("과학 : 70", file=score_file)
score_file.write("음악 : 70\n")
score_file.close()

# score_file = open("score.txt", "r", encoding="utf-8")
# len = score_file.fileno()
# for i in range(1, len+1):
#     print(score_file.readline(), end="")

# score_file = open("score.txt", "r", encoding="utf-8")
# while True:
#     line = score_file.readline()
#     if not line:
#         break
#     print(line, end="")
# score_file.close()
score_file = open("score.txt", "r", encoding="utf-8")
lines = score_file.readlines()
for score in lines:
    print(score, end="")
