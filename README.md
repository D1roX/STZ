Файлы для обучения на колабе для 1-5 попыток (в том числе датасет): https://drive.google.com/drive/folders/10l_GI_3azhkBIWFCjIiIhtX55B4BBvCc
Блокнот 1-5 попыток увы не сохранился, но в целом там был использован макет из ТЗ, ниче не меняли, кроме того, что подгружается с гугл диска


Файлы для обучения на колабе для 6-9 попыток (в том числе датасет): https://drive.google.com/drive/folders/1TnaEPnDIJ4qdBrrpoBPrA-q33YDOOA2V?usp=sharing
Блокнот 6-9 попыток: https://colab.research.google.com/drive/1mVWN4tcyZ-d6JHmKqs_gwQeU6jh8_W5K?usp=sharing

                                        Блок "Ниче не работает"
---------------------------------------------------------------------------------------------------------

            1 ПОПЫТКА                                                                                   
Папка - 1Try                                                                                            
Нейронка - Yolov4, веса недоучились из-за ограничения GPU на колабе                                     
Датасет - 300 размеченных фоток, рандомные фотки ручек из гугла                                         
Результат - ничего не видит                                                                             
                                                                                                        
            2 ПОПЫТКА                                                                                   
Папка - 2Try                                                                                            
Нейронка - Yolov4 (пытались продолжить обучить предыдущие веса (есть такой функционал))                
Датасет - 300 размеченных фоток, рандомные фотки ручек из гугла                                         
Результат - ничего не видит                                                                             
                                                                                                        
            3 ПОПЫТКА                                                                                   
Папка - 3Try                                                                                            
Нейронка - Yolov4 (пытались продолжить обучить веса из 2 попытки)                                       
Датасет - 300 размеченных фоток, рандомные фотки ручек из гугла                                         
Результат - ничего не видит                                                                             
                                                                                                        
            4 ПОПЫТКА                                                                                   
Папка - 4Try                                                                                           
Нейронка - Yolov4 (пытались продолжить обучить веса из 3 попытки)                                       
Датасет - 300 размеченных фоток, рандомные фотки ручек из гугла                                         
Результат - ничего не видит                                                                             
                                                                                                        
            5 ПОПЫТКА                                                                                   
Папка - 5Try                                                                                            
Нейронка - Yolov4 Tiny (первая попытка свапнуть нейронку)                                               
Датасет - 300 размеченных фоток, рандомные фотки ручек из гугла                                         
Результат - ничего не видит                                                                             


                                        Блок "Оно живое"
---------------------------------------------------------------------------------------------------------                                        
                                        
            6 ПОПЫТКА                                                                                   
Папка - 6Try                                                                                            
Нейронка - Yolov4 Tiny (начали обучать с нуля, поигрались побольше с кфг)                               
Датасет - 300 размеченных фоток, создали собственный датасет конкертно 1 ручки                          
Результат - наокнец-то начала распознавать все, что похоже на ручку по форме                            
                                                                                                        
            7 ПОПЫТКА                                                                                   
Папка - 7Try                                                                                            
Нейронка - Yolov4 Tiny (продолжение обучения предыдущих весов)                                         
Датасет - 300 размеченных фоток, наш датасет                                                            
Результат - стала намного лучше видеть границы объекта                                                  
                                                                                                        
            8 ПОПЫТКА                                                                                   
Папка - 8Try                                                                                            
Нейронка - Yolov4 Tiny (продолжение обучения предыдущих весов) - 4к итераций                            
Датасет - 300 размеченных фоток, наш датасет                                                            
Результат - более менее норм распознает ручку, но есть моменты с ошибочным распознаванием других        
            объектов как ручек                                                                          
                                                                                                        
            9 ПОПЫТКА                                                                                   
Папка - 9Try                                                                                            
Нейронка - Yolov4 Tiny (продолжение обучения предыдущих весов) - 5к итераций                            
Датасет - 300 размеченных фоток, наш датасет                                                            
Результат - более менее норм распознает ручку, но есть моменты с ошибочным распознаванием других        
            объектов как ручек. Особо нет отличий от 4к итераций, мб даже хуже                          
