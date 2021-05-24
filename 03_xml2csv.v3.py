# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
  
def xml_to_csv(path):
    xml_list = []
    # 读取注释文件
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
#                     int(root.find('size')[0].text),
#                     int(root.find('size')[1].text),
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     member[0].text
                     )
            xml_list.append(value)
#    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax','class']
  
    # 将所有数据分为样本集和验证集，一般按照4:1的比例
    train_list = xml_list[0: int(len(xml_list)*0.8)]
    val_list = xml_list[int(len(xml_list) * 0.8) + 1: ]
  
    # 保存为CSV格式
    train_df = pd.DataFrame(train_list)
    val_df = pd.DataFrame(val_list)
    train_df.to_csv('./train_anno.csv', index=None,header=0)
    val_df.to_csv('./val_anno.csv', index=None,header=0)
  
  
def main():
    path = './xml'
    xml_to_csv(path)
    print('Successfully converted xml to csv.')
  
main()
