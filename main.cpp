#include <stdio.h>
#include <stdlib.h>

extern void callReduceContiguous();
extern void callReduceInterleaving();

int main(int argc, char **argv) {

//	callReduceContiguous();
	callReduceInterleaving();
}
