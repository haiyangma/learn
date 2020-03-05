import base64
f=open('/2020-03-05/image/mobilenet.png', 'rb')
ls_f=base64.b64encode(f.read())
f.close()
print(ls_f)