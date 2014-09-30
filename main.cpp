#include <stdio.h>
#include <stdlib.h>

// reduce forward declarations
extern void callReduceContiguous();
extern void callReduceInterleaving();

// scan forward declarations
extern void callHellisScan();

int main(int argc, char **argv) {

//	callReduceContiguous();
//	callReduceInterleaving();
	callHellisScan();
}
