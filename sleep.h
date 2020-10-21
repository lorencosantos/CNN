#include <time.h>
#undef nanosleep

#if (defined _WIN32 || defined __WIN64__) && ! defined __CYGWIN__
#include <Windows.h>


int nanosleep(const struct timespec *req, struct timespec *rem)
{
	Sleep((DWORD) (req->tv_sec * 1000)
	      + (DWORD) (req->tv_nsec != 0
			 ? req->tv_nsec / 1000000
			 : 0));

	return 0;
}
#endif
