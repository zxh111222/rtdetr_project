from django import forms

class DetectionForm(forms.Form):
    pt_file = forms.FileField(label='上传权重文件', help_text='请选择权重文件（.pt 或 .pth 格式）')
    image_file = forms.ImageField(label='上传图片文件', help_text='请选择要预测的图像文件')
