TODO
====

1.可以用theano的示例程序，一步进行训练，也可以把卷积的步骤分开，在UI上把卷积的结果显示处理，最后还要把权重也显示出来最好了。
2.调用源码进行训练就行了。


DONE
====
1.将图片进行切分，原图150 * 53, 切成50 * 50的，每张小图以字母_数字命名，小写字母的分隔符为__, 表示这个字母的第多少张图片.
2.WrapAffine数组有40个,目前最多的字母大约25个，那么是1000张，目前总共25000张图片

NOTE
====
1.一共4w张图片左右，52个字母，这样一个字母8000张图片(目标是这样了...)


wrapAffine规则
------------
鉴于最少的字母才5张图片
1.左右旋转20度
2.每次旋转时wrapAffine30次,这样就差不多


python中，像model是package，Image.py是Module，里面定义的class是class，所以要from model.Image as Image, from 一定要定位到Module

'''
下载验证码的网址是
https://ssl.captcha.qq.com/getimage?aid=522205405&r=0.616197673836723&uin=323123232@qq.com
'''
