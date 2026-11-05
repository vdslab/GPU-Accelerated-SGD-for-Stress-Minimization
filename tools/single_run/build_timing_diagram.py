from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
P=Path(__file__).parent
im=Image.new('RGB',(2000,1450),'#fafaf8');d=ImageDraw.Draw(im)
font='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'; bold='/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'
def text(x,y,s,size=26,color='#303a40',heavy=False):
 d.text((x,y),s,font=ImageFont.truetype(bold if heavy else font,size),fill=color)
def center(x,y,s,size=24,color='#303a40'):
 f=ImageFont.truetype(font,size);w=d.textbbox((0,0),s,font=f)[2];d.text((x-w/2,y),s,font=f,fill=color)
blue='#52677f'; purple='#4c8077'; pale='#e8efec'
xs=[380,580,760,950,1480,1650,1900]
text(50,28,'実行時間の計測区間',44,heavy=True)
text(50,90,'説明用に2反復へ省略。横幅は実測時間ではありません。',25,color='#727b80')
labels=[('A','実行開始'),('B','前処理完了'),('C','GPU資源','準備完了'),('D','初期転送','完了'),('E','読戻し','開始直前'),('F','CPU座標','取得完了'),('G','描画完了')]
for x,ls in zip(xs,labels):
 center(x,145,ls[0],27,blue)
 for j,t in enumerate(ls[1:]):center(x,185+j*33,t,22)
 d.line((x,255,x,1180),fill='#dce1df',width=2)
phases=['前処理','GPU準備','初期転送','反復処理','読戻し','描画準備・描画']
for i,label in enumerate(phases):
 a,b=xs[i],xs[i+1]; d.rounded_rectangle((a+3,260,b-3,312),8,fill='#e8ecef');center((a+b)/2,270,label,22)
text(50,340,'CPUの先行生成',25)
d.rounded_rectangle((950,337,995,371),5,fill='#a58e74')
d.rounded_rectangle((1020,337,1160,371),5,fill='#a58e74')
text(1015,380,'反復1のGPU計算中に、順列2を生成',20,color='#776651')
text(50,415,'GPUの計算区間',25)
segments=[(1000,1080),(1100,1160),(1240,1320),(1340,1420)]
for a,b in segments:d.rounded_rectangle((a,416,b,449),5,fill=purple)
text(50,475,'線：区間全体の経過時間　　塗り：GPU計算区間の合計（隙間は除く）',25,heavy=True)
def bar(a,b,y,color=blue):
 d.line((a,y,b,y),fill=color,width=9)
 for x in [a,b]:d.line((x,y-13,x,y+13),fill=color,width=4)
def label(n,name,y):
 text(50,y-22,n+' '+name,24,heavy=True)
for n,name,a,b,y in [('①','前処理時間',380,580,555),('②','GPU準備時間',580,760,620),('③','初期転送時間',760,950,685)]:
 label(n,name,y);bar(a,b,y)
label('④','反復時間',765)
for a,b in segments:d.rounded_rectangle((a,749,b,777),5,fill=purple)
center(1080,790,'反復1の計算区間',22,purple);center(1330,790,'反復2の計算区間',22,purple)
for n,name,a,b,y in [('⑤','SGD実行時間',950,1480,860),('⑥','結果取得時間',1480,1650,925),('⑦','描画時間',1650,1900,990),('⑩','総計算時間',380,1650,1055),('⑧','総計算時間（描画込み）',380,1900,1120)]:
 label(n,name,y);bar(a,b,y)
label('⑨','反復時間合計',1210)
for a,b in segments:d.rounded_rectangle((a,1194,b,1222),5,fill=purple)
center(1210,1236,'反復1の④ ＋ 反復2の④',25,purple)
d.rounded_rectangle((45,1290,1950,1420),12,fill='#e9edeb')
text(70,1305,'⑤ SGD実行時間（反復時間合計＋GPU計算外の経過時間）',29,heavy=True)
text(70,1348,'GPU計算区間内のメモリ待ち・WGバリアを含む。CPU処理時間は加算しない。',23)
text(70,1383,'⑩：A〜F（CPU座標取得まで）　⑧：A〜G（最終配置のGPU描画完了まで）',23)
im.save(P/'timing-diagram.png')
