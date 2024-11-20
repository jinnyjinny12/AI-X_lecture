# Quiz) 주어진 코드를 가지고 부동산 프로그램을 작성하세요.

# (출력 예제)
# 총 3대의 매물이 있습니다.
# 강남 아파트 매매 10억 2010년
# 마포 오피스텔 전세 5억 2007년
# 송파 빌라 월세 500/50 2000년


class house:
    def __init__(self, location, house_type, deal_type, price, completion_year):
        self.location = location
        self.house_type = house_type
        self.deal_type = deal_type
        self.price = price
        self.completion_year = completion_year

    def show_detail(self):
        return "{0} {1} {2} {3} {4}"\
            .format(self.location,\
                    self.house_type,\
                    self.deal_type,\
                    self.price,\
                    self.completion_year)
    
def show_house_list(house_list:list[house]):
    size = len(house_list)
    print("총 {}대의 매물이 있습니다.".format(size))
    for house in house_list:
        print(house.show_detail())

house_list = [house("강남", "아파트", "매매", "10억", "2010년"), \
            house("마포", "오피스텔", "전세", "5억", "2007년"), \
            house("송파", "빌라", "월세", "500/50", "2000년")]


show_house_list(house_list)


