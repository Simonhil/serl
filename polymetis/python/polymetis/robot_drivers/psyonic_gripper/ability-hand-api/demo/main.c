#include "i2c-master-test.h"
#include <signal.h>
#include <string.h>
#include "i2c-err-lookup.h"

float current_time_sec(struct timeval * tv)
{
	gettimeofday(tv,NULL);
	int64_t t_int = (tv->tv_sec*1000000+tv->tv_usec);
	return ((float)t_int)/1000000.0f;
}

static volatile int gl_leave_loop = 0;
void int_handler(int tmp)
{
	gl_leave_loop = 1;
}

uint16_t get_max(uint16_t * list, int listsize)
{
	uint16_t max = 0;
	for(int i = 0; i < listsize; i++)
	{
		if(list[i] > max)
			max = list[i];
	}
	return max;
}


void main()
{
	signal(SIGINT, int_handler);

	open_i2c(0x50);	//Initialize the I2C port. Currently default setting is 100kHz clock rate

	/*Quick example of pre-programmed grip control (i.e. separate control mode from torque, velocity and position control)*/
	// set_grip(GENERAL_OPEN_CMD,100);
	// usleep(3000000);

	// set_grip(PINCH_GRASP_CMD,100);
	// usleep(3000000);

	// set_grip(SIGN_OF_THE_HORNS_GRASP,200);
	// usleep(3000000);

	// set_grip(RUDE_POINT_CMD,200);
	// usleep(3000000);

	// set_grip(OPEN_KEY_CMD,200);
	// usleep(3000000);

	// set_grip(GENERAL_OPEN_CMD,250);
	// usleep(3000000);

	/*Setpoint generation start time*/
	struct timeval tv;

	/*All control modes will use the same float format struct for input and output. Initializing them here*/
	float_format_i2c i2c_out;
	for(int ch = 0; ch < NUM_CHANNELS; ch++)
		i2c_out.v[ch] = 0;
	float fpos[NUM_CHANNELS] = {0};
	float iq[NUM_CHANNELS] = {0};

	pres_fmt_i2c pres_fmt[NUM_CHANNELS] = {0};

	set_mode(POS_CTL_MODE);

	/*Setup for demo motion*/
	uint8_t disabled_stat = 0;

	float start_ts = current_time_sec(&tv);
	float test_config[NUM_CHANNELS] = {15.f,15.f,15.f,15.f,15.f,-80.f};
	//printf("\033[2J\033[1;1H");
	char buffer[4096*4] = {0};
	int calls=0;

	int errors=0;
	while(gl_leave_loop == 0)
	{

		float t = current_time_sec(&tv) - start_ts;

		/*
		Pressure Indices:
		Index: 	0-3
		Middle: 4-7
		Ring: 	8-11
		Pinky: 	12-15
		Thumb: 	16-19

		Note that the The pressure range is NOT normalized (i.e. will range from 0-0xFFFF).
		*/
		//float ft = t*1.f;
		float ft = 1.f;
		for(int ch =0; ch < NUM_CHANNELS; ch++)
		{
			if(ch != THUMB_ROTATOR)
				i2c_out.v[ch] = 30.f*(-cos(ft+(float)ch)*.5f+.5f)+15.f;
			else if (ch == THUMB_ROTATOR)
				i2c_out.v[ch] = 30.f*(-cos(ft+(float)ch)*.5f+.5f)-80.f;
		}

		calls++;
		//int rc = api_frame_fmt_2(POS_CTL_MODE, &i2c_out, fpos, iq, &disabled_stat, pres_fmt);
		int rc = api_frame_fmt_1(POS_CTL_MODE, &i2c_out, fpos, &disabled_stat, pres_fmt);	//current default option. default will be replaced with fmt_2 in future release
		//int rc = api_frame_fmt_1(READ_ONLY_MODE, &i2c_out, fpos, &disabled_stat, pres_fmt);	//current default option. default will be replaced with fmt_2 in future release
		
		if(rc != 0) {
			errors++;			
			print_hr_errcode(rc);
		}
		else
		{
			const char * name[NUM_CHANNELS] = {"index", "middle","ring","pinky","thumb flexor", "thumb rotator"};
			//printf("\033[2J\033[1;1H");

			if(0)  {

				int length = 0;
				for(int finger = 0; finger < 6; finger++)
				{
					length += sprintf(buffer+length, "%s ", name[finger]);
					//length += sprintf(buffer+length, "--------------------------------------------------\r\n");
					if(finger < 5) //there is no pressure sensor on the thumb rotator
					{
						length += sprintf(buffer+length, "sensor: [");
						for(int sensor = 0; sensor < 6; sensor ++)
						{
							length += sprintf(buffer+length, "%.4d, ", pres_fmt[finger].v[sensor]);
						}
						length += sprintf(buffer+length, "]                             \r\n");					
					}
					length += sprintf(buffer+length, "fingerpos: %.2f, current: %.2f                                        \r\n", fpos[finger], iq[finger]);
					length += sprintf(buffer+length, "--------------------------------------------------\r\n");
					//length += sprintf(buffer+length, "                                                           \r\n");
					printf("%s", buffer);
				}
			} else {
				char *p = buffer;
				int i = 0;
				// header row
				p += sprintf(p, "           " );
				for(i=0; i<NUM_CHANNELS; i++) {
					p += sprintf(p, "%-15s | ", name[i]);
				}
				p += sprintf(p, "%-15s\n", "Calls/Errors");

				// position, errors
				p += sprintf(p, "Position | " );
				for(i=0; i<NUM_CHANNELS; i++) {
					p += sprintf(p, "%15.4f | ", fpos[i]);
				}
				p += sprintf(p, "%d / %d\n", calls, errors);

				// current
				p += sprintf(p, "Current  | " );
				for(i=0; i<NUM_CHANNELS; i++) {
					p += sprintf(p, "%15.4f | ", iq[i]);
				}
				p += sprintf(p, "\n");


				// pressure data
				for(int sensor=0; sensor<6; sensor++) {
					p += sprintf(p, "%s %d | ", "sensor", sensor );
					for(i=0; i<NUM_CHANNELS; i++) {
						p += sprintf(p, "%15.4d | ", pres_fmt[i].v[sensor]);
					}
					p += sprintf(p, "\n");
				}

				printf("%s", buffer);
			}

		}
		usleep(100000);
	}
	printf("Exit Program\r\n");

}

