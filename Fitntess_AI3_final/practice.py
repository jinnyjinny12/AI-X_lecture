from random import *

def introduce(animal: str, name:str, age:int, hobby: str):
    is_adult = age > 3

    print("우리집 "+ animal+ "의 이름은 " + name + "에요")
    print(name + "는 "+ str(age)+ "살이며 " + hobby + "을(를) 좋아해요.")
    print(name + "는 어른일까요? " + str(is_adult))

introduce("강아지", "해피", 3, "산책")

def operator(x: int, y: int):
    sum = x + y
    subtraction = x - y
    divide = x / y
    multipy = x * y
    square_root = x ** y
    rest_num = x % y
    share = x // y
    print("x + Y = " , sum)
    print("x - Y = " , subtraction)
    print("x / Y = " , divide)
    print("x * Y = " , multipy)
    print("x ** Y = " , square_root)
    print("x % y = ", rest_num)
    print("x // y = ", share)

operator(6, 2)

def random_online_class_date(date: int):
    day = randrange(4, date)
    print("오프라인 스터디 모임은 매월 " + str(day) + "로 정해졌습니다.")

random_online_class_date(28)

def string_slice(jumin: str):
    sex = "남" if jumin[7] == "1" else "여"
    year = jumin[0:2]
    month = jumin[2:4]
    day = jumin[4:6]
    print("성별 : " + sex)
    print("생년월일 : "+ year+"월" + month + "월" + day +"일")
    print("주민번호 앞자리 : " + jumin[:6])
    print("주민번호 뒷자리 : " + jumin[7:])
    print("주민번호 뒷자리를 가져옴(뭔자열 뒤에서부터 시작)" + jumin[-7:])

string_slice("991108-1122334")


def string_upper_and_lower(python:str) :
    print(python.lower())
    print(python.upper())
    print(python[0].isupper())
    print(len(python))
    print(python.replace("Python", "Java"))
    index = python.index("n")
    print("n -> index  : " + str(index))
    print("n에 대한 다음 인덱스 : " + str(python.index("n", index+1)))
    print("find(Python) : " + str(python.find("Python")))
    print("find(Java) : " + str(python.find("Java")))
    print("count(n) : " + str(python.count("n")))
          
string_upper_and_lower("Python is Amazing")

def string_format():
    print("나는 %d살 입니다." % 20)
    print("나는 %s살 입니다." % 20)
    print("나는 %s을 좋아합니다." % "python")
    print("Apple은 %c로 시작해요" % "A")
    print("나는 %s색과 %s색을 좋아해요." % ("파랑", "보라"))
    print("나는 {}이고 {}살입니다.".format("박준현", 20))
    print("나는 {1}이고 {0}살입니다.".format("박준현", 20))
    print("나는 {name}이고 {age}살입니다.".format(name="배한실", age=33))
    #version 3.6~
    name="박준현"
    age="34"
    print(f"나는 {name}이고 {age}살입니다.")

string_format()

def escape_string():
    print("백문이 불여일견 \n백견이 불여일타")  
    print("나는 \"나도코딩\" 입니다.")
    # \r : 맨앞으로 이동해서 같은 길이의 문자열로 덮어쓰기 됨.
    print("red apple \rpine")
    # \b : 백스페이스
    print("안녕이다 ㅈ\b ")
    print("redd\b apple")
    # \t
    print("\t안녕")

escape_string()

def make_site_password(url:str):
    domain = url[url.index(".") + 1: url.index(".", url.index(".") + 1)]
    password = "{}{}{}{}".format(domain[:3], len(domain), domain.count("e"), "!")
    
    return password

password = make_site_password("http://www.daum.net")

print(password)

def list_util():
    def list_extend(arr1:list, arr2:list):
        arr1.extend(arr2)
        return arr1

    def add_and_print(station_list:list, station:str):
        station_list.append(station)
        print(station_list)

    def pop_and_print(station_list: list):
        print(station_list.pop())
        print(station_list)

    num_list = [5,2,3,1,4]
    num_list.sort()
    print(num_list)
    num_list.reverse()
    print(num_list)
    station_list = []
    add_and_print(station_list, "신도림")
    add_and_print(station_list, "대림")
    add_and_print(station_list, "구로디지털단지")
    add_and_print(station_list, "신대방")
    pop_and_print(station_list)
    pop_and_print(station_list)

    mix_list = list_extend(num_list, station_list)
    print(mix_list)

    
list_util()



