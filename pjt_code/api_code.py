# 2018-10-24 공공데이터포털 아파트실거래 상세정보 api 읽어옴, xml 파일 csv 변환 시도

from datetime import datetime
from dateutil.relativedelta import relativedelta
import urllib.request
import xml.etree.ElementTree as ET
import csv

# LAWD_CD - 지역코드
# DEAL_YMD - 계약 년월
# 거래금액 건축년도 년 법정동 아파트 월 일 전용면적 지번 지역코드 층

# 원하는 기간 입력(시작, 끝) 후 해당 데이터 csv파일로 변환해 저장하는 함수 작성
def apartment_detail_until(lawd_cd, start_ymd, end_ymd):
    api_key = '5l05HUL1Miw0FSLIj3fPxGMww8FZrTGXOsx' \
              'VQ69wfpeVJypWPowtOoozOX0BGJYxElx6ZEEDKrFl2XSaD%2F1NZg%3D%3D'  # UTF-8 본인 계정에서 받은 api 키
    api_url = 'http://apis.data.go.kr/1611000/AptBasisInfoService'

    start_date = datetime.strptime(str(start_ymd), '%Y%m').date()
    end_date = datetime.strptime(str(end_ymd), '%Y%m').date()
    period = relativedelta(end_date, start_date)
    period_month = int(period.months) + int(period.years) * 12 + 1
    # ---api url 처리, 데이터 구하는 기간 계산 ---

    csv_file = open(str(start_ymd) + '-' + str(end_ymd) + 'data.csv', 'w', newline='')
    csvwriter = csv.writer(csv_file)
    # csv 파일 작성용 writer

    for i in range(0, period_month):
        per_mon = relativedelta(months=i)
        date1 = start_date + per_mon
        date1 = datetime.strftime(date1, '%Y%m')
        url = api_url + '?LAWD_CD=' + str(lawd_cd) + '&DEAL_YMD=' + str(date1) + '&serviceKey=' + api_key
        api_data = urllib.request.urlopen(url).read().decode("utf-8")
        root = ET.fromstring(api_data)
        elements = root.findall('body/items/item')

        for idx, item in enumerate(elements):
            col_list = []
            item_list = []
            if i == 0 and idx == 0:
                col_list.append('계약년월')
                col_list.append(item.find('지역코드').tag)
                col_list.append(item.find('거래금액').tag)
                col_list.append(item.find('건축년도').tag)
                col_list.append(item.find('법정동').tag)
                col_list.append(item.find('년').tag)
                col_list.append(item.find('월').tag)
                col_list.append(item.find('일').tag)
                col_list.append(item.find('지번').tag)
                col_list.append(item.find('아파트').tag)
                col_list.append(item.find('전용면적').tag)
                col_list.append(item.find('층').tag)
                csvwriter.writerow(col_list)  # 컬럼 생성용

            item_list.append(date1)
            item_list.append(item.find('지역코드').text)
            item_list.append(item.find('거래금액').text)
            item_list.append(item.find('건축년도').text)
            item_list.append(item.find('법정동').text)
            item_list.append(item.find('년').text)
            item_list.append(item.find('월').text)
            item_list.append(item.find('일').text)
            item_list.append(item.find('지번').text)
            item_list.append(item.find('아파트').text)
            item_list.append(item.find('전용면적').text)
            item_list.append(item.find('층').text)
            csvwriter.writerow(item_list)  # 데이터

    csv_file.close()

apartment_detail_until(11110, 201501, 201809)
# 페이지 번호 / 시작 페이지 / 열 갯수 / 메이지 사이즈 / 지역코드 / 계약 년월