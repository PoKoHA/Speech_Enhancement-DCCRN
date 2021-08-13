## DCCRN: Deep Complex Convolution Recurrent Network
* Paper:  https://arxiv.org/abs/2008.00264
* Official Code: https://github.com/huyanxin/DeepComplexCRN

**- Network**

![arch](https://user-images.githubusercontent.com/76771847/129316005-1c19ff9a-dd37-43e6-982e-0b1dd298acc9.png)
>DCUNet 와 비슷한 형태로 다른 점은 가운데 Bridge 부분에 Complex LSTM이 있다.
또한 STFT을 Conv1d, ISTFT을 ConvTranspose1d 을 사용하였다.

**- Module**

![module](https://user-images.githubusercontent.com/76771847/129316229-51ca7c51-c058-40f4-a1f5-80d47e32a236.png)
> Real과 Imag 을 Stack(or Concat)으로 뭉쳐서 Conv(or Batch, ConvTranspose) 실행 할 때 다시 chunk(or Slicing)으로 풀어 실행
> 
> Complex Multiplication 식과 동일하게 수행

**- Loss**

![loss](https://user-images.githubusercontent.com/76771847/129316595-bad11735-78e6-45e1-8fa5-fe67e422423f.png)


**- Mask Type**

![d](https://user-images.githubusercontent.com/76771847/129335884-bdccfea9-05c5-435d-92f6-96bc40124c44.png)

## Reference

code: https://github.com/stdKonjac

code: https://github.com/seorim0/DCCRN-with-various-loss-functions

DCUnet code

