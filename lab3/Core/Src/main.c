
/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <stdio.h>
#include <math.h>
#include "config.h"
#include "fixedpt.h"

/**
  * @brief  The application entry point.
  * @retval int
  */
	
#define LUT_SIZE 628

float sinLUT[LUT_SIZE];

float generateSinNx(int n, float x) {
    int index = (int)((x / (2 * 3.14)) * LUT_SIZE) % LUT_SIZE;
    if (index < 0) {
        index += LUT_SIZE;  // Handle negative indices
    }
	//sin(n*x) 
    return sinLUT[index * n];
}

void generateSquareWave(int numTerms) {
	//__HAL_TIM_SET_COUNTER(&htim6, 0);
	//float t0 =__HAL_TIM_GET_COUNTER(&htim6);
		int p = 0;
		for (float i = 0; i < (2 * 3.14); i += ((2 * 3.14)/LUT_SIZE)) {
        float squareWave = 0.5;  
				float sum = 0;
				for (int n = 0; n < numTerms; n++) {
					sum += (1.0 / (2* n + 1)) * generateSinNx(2*n+1, i);
				}
				squareWave+= (sum * (2/3.14));
				
				if(p % p == 0){
					char p3msg[100];
					sprintf(p3msg, "square wave val: %f\n", squareWave);
					//print_msg(p3msg);
				}
				p++;
				int dacValue = 0;
				if (!isnan(squareWave) && squareWave <= 5) {
					dacValue = (int)((squareWave + 1.0) * 2048);
					HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, dacValue);
				}
				else {
					dacValue = (int)((2.0 + 1.0) * 2048);
					HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, dacValue);
				}
				
		}
		//REMEMBER TO SET DAC SOMEWHERE HERE OR IN FUNCTION, IF IN FN CHANGE TO RETURN TYPE FLOAT AND RETURN squareWave
}
void generateSawToothWave(int numTerms) {
		int p = 0;
//	__HAL_TIM_SET_COUNTER(&htim6, 0);
	//float t0 =__HAL_TIM_GET_COUNTER(&htim6);
		for (float i = 0; i < (2 * 3.14); i += ((2 * 3.14)/LUT_SIZE)) {
        float sawWave = 0.5;  
				float sum = 0;
				for (int n = 1; n < numTerms; n++) {
					sum += (1.0/n) * generateSinNx(n, i);
				}
				sawWave -= ((1/3.14)* sum);
				if(p % p == 0){
					char p3msg[100];
					sprintf(p3msg, "saw wave val: %f\n", sawWave);
					//print_msg(p3msg);
				}
				p++;
				int dacValue = 0;
				if (!isnan(sawWave) && sawWave <= 5) {
					dacValue = (int)((sawWave + 1.0) * 2048);
					HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, dacValue);
				}
				
				
		}
		//REMEMBER TO SET DAC SOMEWHERE HERE OR IN FUNCTION, IF IN FN CHANGE TO RETURN TYPE FLOAT AND RETURN squareWave
}
void generateTriangularWave(int numTerms) {
		int p = 0;
	//__HAL_TIM_SET_COUNTER(&htim6, 0);
	//float t0 =__HAL_TIM_GET_COUNTER(&htim6);
		for (float i = 0; i < (2 * 3.14); i += ((2 * 3.14)/LUT_SIZE)) {
        float triWave = 0.5;  
				float sum = 0;
				for (int n = 0; n < numTerms; n++) {
					sum += (pow(-1,n)/pow((2*n+1),2)) * generateSinNx(2*n+1, i);
				}
				triWave += ((4/pow(3.14,2))* sum);
				//you can use this to control how much is printed
				if(p % p == 0){
					char p3msg[100];
					sprintf(p3msg, "triangle wave val: %f\n", triWave);
					//print_msg(p3msg);
				}
				p++;
				int dacValue = 0;
				if (!isnan(triWave) && triWave <= 5) {
					dacValue = (int)((triWave + 1.0) * 2048);
					HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, dacValue);
				}
		}
		
		//REMEMBER TO SET DAC SOMEWHERE HERE OR IN FUNCTION, IF IN FN CHANGE TO RETURN TYPE FLOAT AND RETURN squareWave
}





int numTerms;
int main(void)
{
  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DAC_Init();
  MX_TIM6_Init();
  MX_USART3_Init();
	HAL_DAC_Init(&hdac);
  HAL_DAC_Start(&hdac, DAC_CHANNEL_1);
	
		__HAL_TIM_SET_PRESCALER(&htim6, 1023);
	HAL_TIM_Base_Start(&htim6);
		
	
	char message[20];
	sprintf(message, "New Execution.\n");
	print_msg(message);
	
	char rollover[20];
	sprintf(rollover, "Rolled Over.\n");
	
	
	int numTerms = 0;
  while (1)
  {
		
		//part1
		float inc = (2*3.14)/50;
		
		HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
		/*
		float t0 =__HAL_TIM_GET_COUNTER(&htim6);
		int j = 0;
		for(int i=0; j<100 ; i++){
			j++;
		}
		float t1 =__HAL_TIM_GET_COUNTER(&htim6);
		
		if (t1<t0){
			print_msg(rollover);
		}
		float time= (t1 - t0);
		char output[100];
		sprintf(output,"The time taken to sum to %d: %f\n",j,time);
		print_msg(output);
		
		
		
		
		//part2
		
		
		//HAL_TIM_Base_Stop(&htim6);
		//reset counter test//
		//__HAL_TIM_SET_COUNTER(&htim6, 0);
		//HAL_TIM_Base_Start(&htim6);
		float out;
		
		float t3 = __HAL_TIM_GET_COUNTER(&htim6); 
		for(float rad = 0.00; rad < (2*3.14); rad+=inc){
			out = sin(rad);
			float dacVal = (out * 4095)/4095;
			HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, dacVal);
		
			///test///
				
			char test[50];
			sprintf(test, "value: %f\n",dacVal);
			print_msg(test);
			HAL_Delay(100);
		}
		
		////test////
		
		float t4 = __HAL_TIM_GET_COUNTER(&htim6);
		///HAL_TIM_Base_Stop(&htim6);
		int valid=1;
		if(t4<t3){
			print_msg(rollover);
			valid=0;
		}
			if(valid){
		float time_taken = (t4 - t3);
		
		char part2[100];
		//f = 1/T
		float frequency = 1.0/(time_taken/(float)(SystemCoreClock/2));
		sprintf(part2, "Frequency: %f Hz, Timer Cycles, Core clock %d: %f\n", frequency, SystemCoreClock, time_taken);
		print_msg(part2);
			}
		
			
		//part2.2
	
		__HAL_TIM_SET_COUNTER(&htim6, 0);
		char part2_2[100];
		
		float LUT[50];
		int k=0;
		
			for(float rad = 0.00; rad < (2*3.14); rad+=inc){
			out = sin(rad);
			LUT[k]=out;
			k++;
			}
			
		//get only lut output time not allocation as well	
		float t5 = __HAL_TIM_GET_COUNTER(&htim6);
			for (int i =0; i<k; i++){
				float dacVal = (LUT[i] * 4095)/4095;
				HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, dacVal);
				//char lut[100];
				//sprintf(lut,"Sin Value DAC: %f\n",LUT[i]);
				//print_msg(lut);
			}
		float t6 = __HAL_TIM_GET_COUNTER(&htim6);
		float time_taken = (t6 - t5);	
		float frequency = 1.0/(time_taken/(float)(SystemCoreClock/2));
		sprintf(part2_2, "Frequency: %f Hz, Timer Cycles, Core clock %d: %f\n", frequency, SystemCoreClock, time_taken);
		print_msg(part2_2);
		
		*/
		
		
			
		//part 3 generating square wave
		 // GET SIN LUT
		for (int i = 0; i < LUT_SIZE; ++i) {
			float rad = (2 * 3.14 / (LUT_SIZE -1)) * i;
			sinLUT[i] = sin(rad);
		}
		
		
		
		if (HAL_GPIO_ReadPin(LD1_GPIO_Port,LD1_Pin) == GPIO_PIN_SET){
			numTerms+=1;
			HAL_GPIO_WritePin(LD1_GPIO_Port,LD1_Pin, GPIO_PIN_RESET);
		}
		int num = 0;
		
		char numT[20];
		sprintf(numT, "numterms is %d\n", numTerms);
		print_msg(numT);
		
		//generateSquareWave(numTerms);
		//generateSawToothWave(numTerms);
		generateTriangularWave(numTerms);
		
		
		//part 4
		/*
		fixedpt sin_values[LUT_SIZE];
		int k=0;
		//float out=0.0;
		__HAL_TIM_SET_COUNTER(&htim6, 0);
		float t7 = __HAL_TIM_GET_COUNTER(&htim6);
		for(float rad = 0.00; rad < (2*3.14); rad+=inc){
			//out = sin(rad);
			sin_values[k]=FXD_FROM_FLOAT(sin(rad));
			//test//
			char kamini[50];
			fixedpt Dacval = ((sin_values[k] +1.0)/2048);
			int pass = FXD_TO_INT(Dacval);
			sprintf(kamini,"Value: %d\n",sin_values[k]);
			print_msg(kamini);
			HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, pass);
			
			k++;
			
		}
		float t8 = __HAL_TIM_GET_COUNTER(&htim6);
		
		float time_taken = (t8 - t7);
		char part4[100];
		sprintf(part4, "\n\n\ntime taken: %f\n\n\n", time_taken );
		print_msg(part4);
		*/
		

		
		
  }
}