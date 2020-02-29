/* widthA=heightB for valid matrix multiplication */
__kernel void simpleMultiply(
    __global float *outputD,
    int widthA,
    int heightA,
    int widthB,
    int heightB,
    int widthC,
    int heightC,
    int widthTmp,
    int heightTmp,
    __global float *inputA,
    __global float *inputB,
    __global float *inputC,
    __global float *inputTmp)
{
    /* get global position in Y direction */
    int row = get_global_id (1);
    /* get global position in X direction */
    int col = get_global_id (0);
    
    int row1 = get_global_id(1);
    int col1 = get_global_id(0);

    float sum = 0.0f;

    /* calculate result of one element of Matrix C */
    for (int i=0; i<widthA; i++) {
        sum += inputA[row*widthA + i] * inputB[i*widthB + col];
    }
    inputTmp[row*widthB + col] = sum;
    float sum1 =0.0f;
    for(int j = 0; j<widthC; j++)
    {
      sum1 = inputC[row1*widthC + col] + inputTmp[row1*widthTmp + col1];
    }
    outputD[row1*widthC + col1] = sum1;
}
