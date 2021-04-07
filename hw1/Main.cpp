#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
	auto experiments = 10000000;
	// auto x = rand();

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < experiments; ++i) {
		int k = pow(i, 2);
	}
	auto powTime = std::chrono::high_resolution_clock::now() - start;
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < experiments; ++i) {
		int k = i * i;
	}
	auto mulTime = std::chrono::high_resolution_clock::now() - start;
	std::cout << "pow time " << std::chrono::duration_cast<std::chrono::milliseconds>(powTime).count() << std::endl;
	std::cout << "mul time " << std::chrono::duration_cast<std::chrono::milliseconds>(mulTime).count() << std::endl;
}


/*
I am running the experiments 10000000 times just so the results are more stable.

taking the square of a random constant again and again took the following times (milliseconds):
pow function time = 1390
time to multiply the number with itself = 30
In this case, the pow function is 46.33 worse than just multiplying the number with itself.

taking the square of the numbers from 0 to 9999999 took the following times (milliseconds):
pow function time = 673
time to multiply the number with itself = 19
In this case, the pow function is 51.21 worse than just multiplying the number with itself.

I tried for a lot of cases and for a lot of experiments but almost every time multiplication
was 40-50 times better than using the pow function, except for the cases when the for loop
consisted of some more work (in those cases the outcome wasn't as drastic but multiplication
was still a lot faster than the pow function).
I think that this happens because the pow function is a more general-purpose function, and
it would have its own input checks and logic, and all that would result in quite a few lines
of code when converted to assembly, whereas the multiplication would just be a single line of
code (or at least way fewer instructions than the pow function) in the assembly.
So, I don't think it ever makes sense to use the pow function to calculate the square of a
number, at least for the cases when we're not only concerned about the runtime complexity but
also the exact runtime.
*/


