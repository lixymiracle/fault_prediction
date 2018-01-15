import csv
import xlrd
import xlwt

with open('/home/lixiangyu/Desktop/20003001#2017-03.csv', 'rt', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for i, rows in enumerate(csv_reader):
        if i == 0:
            row = rows

xlsxfile = xlrd.open_workbook(r'/home/lixiangyu/Desktop/2MWFeatureList.xlsx')
sheet = xlsxfile.sheet_by_index(0)
col1 = sheet.col_values(0)
col2 = sheet.col_values(1)

workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('cata')

try:
    for r in row:
        worksheet.write(0, row.index(r), label=r)
        if r in col1:
            index = col1.index(r)
            worksheet.write(1, row.index(r), label=col2[index])
        else:
            worksheet.write(1, row.index(r), label='null')

except:
    print("error")

workbook.save('cata.xls')