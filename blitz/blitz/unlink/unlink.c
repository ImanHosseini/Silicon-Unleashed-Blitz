#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>

// // Only works on Linux
// char *recover_filename(FILE *f)
// {
//     char fd_path[256];
//     int fd = fileno(f);
//     sprintf(fd_path, "/proc/self/fd/%d", fd);
//     char *filename = malloc(256);
//     int n;
//     if ((n = readlink(fd_path, filename, 255)) < 0)
//         return NULL;
//     filename[n] = '\0';
//     return filename;
// }

// size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
// {
//     size_t (*lfread)(void *, size_t, size_t, FILE*) = dlsym(RTLD_NEXT, "fread");
//     char *fname = recover_filename(stream);
//     printf("Read from file %s\n", fname);
//     free(fname);
//     return lfread(ptr, size, nmemb, stream);
// }

// #cc -fPIC -shared -o myunlink.so myunlink.c -ldl
// #LD_PRELOAD=./unlk/myunlink.so gcc h.c
#include <string.h>

int unlink(const char *name){
    int (*_unlink)(const char*) = dlsym(RTLD_NEXT, "unlink");
    int i = 1;
    char c0 = name[0];
    char c1 = name[1];
    while (name[++i] != '\0'){
        c0 = c1;
        c1 = name[i];
    }
    printf("%s | %c | %c\n",name,c0,c1);
    if(c1 == 's' && c0 == '.'){
        printf("*");
        return 0;
    }
    printf(".");
    return _unlink(name);
}

