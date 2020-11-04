#ifndef LIBHBT_H
#define LIBHBT_H

void hbt_init(char *config_file, int num_threads);
void hbt_invoke(int first_snapnum, int this_snapnum);
void hbt_free(void);

#endif
