#ifndef __EXAMPLE_UTILS_H__
#define __EXAMPLE_UTILS_H__

static int get_int_param(
    const int argc, char **argv, const int argi, 
    const int default_val, const int min_val, const int max_val)
{
    int res = default_val;
    if (argc >= argi + 1) 
    {
        res = atoi(argv[argi]);
        if (res < min_val || res > max_val) res = default_val;
    }
    return res;
}

#endif
