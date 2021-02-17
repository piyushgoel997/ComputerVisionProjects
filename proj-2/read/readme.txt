OS used - Windows 10
IDE - Visual Studio 19 with Visual C++

Simply compile the code (with opencv and C++17) by keeping all the .cpp and .h files in one folder.

The command to use the program after compilation is as follows:
<program executeable name> <T> <B> <F> <D> <N> <F_db>

<program name>: The name of the program executable.
<T>: This would either be the name of the image file or it would be the character i. The character i would launch interactive mode (helps run cbir for multiple images).
<B>: Absolute path of the image database. (IMP: Don't forget to add a backslash to the end of the path.)
<F>: Name of the matching method.
<D>: Name of the distance metric.
<N>: Number of top matches to be displayed.
<F_db>: Optional argument - Absolute path of the databse where the feature files for each image is (or will be) stored. In case this argument is omitted, then the program would create a temporary directory in the program directory to store the features, which will be deleted if the program quits normally.(Warning: Provifing the path to a different matcher's feature database would result in undefined behavior.) (IMP: Don't forget to add a backslash to the end of the path.)


Possible values of <F> and the matching algorithm they correspond to:
'b' or "baseline": Baseline matching algorithm
'h-32' or "histogram-32": RG-chromaticity 2d-histogram matching algorithm with 32 buckets.
'h' or "histogram": RG-chromaticity 2d-histogram matching algorithm with n (>0) buckets, where n would be taken as input from the user.
'mh-tb-32' or "multihistogram-topbottom-32": RG-chromaticity 2d-histogram matching algorithm with two separate histograms for the top half and the bottom half of the image with 32 buckets.
'mh-tb' or "multihistogram-topbottom": RG-chromaticity 2d-histogram matching algorithm with two separate histograms for the top half and the bottom half of the image with n (>0) buckets, where n would be taken as input from the user.
'mh-cf-32' or "multihistogram-centerfull-32": RG-chromaticity 2d-histogram matching algorithm with two separate histograms for the center part and the complete image with 32 buckets.
'mh-cf' or "multihistogram-centerfull": RG-chromaticity 2d-histogram matching algorithm with two separate histograms for the center part and the complete image with n (>0) buckets, where n would be taken as input from the user.
'h-s-32' or "histogram-sobel-32": A combination of RG-chromaticity 2d-histogram on the full image and sobel orientation texture on the full image, both with 32 buckets.
'h-s' or "histogram-sobel": A combination of RG-chromaticity 2d-histogram on the full image and sobel orientation texture histogram on the full image, both with n (>0) buckets, where n would be taken as input from the user.
'mh-cf-ms-tb-32' or "multihistogram-centerfull-multisobel-topbottom-32": A combination of multihistogram-centerfull and a sobel orientation texture histogram on the top half and one on the bottom half of the image, all three with 32 buckets.
'mh-cf-ms-tb' or "multihistogram-centerfull-multisobel-topbottom": A combination of multihistogram-centerfull and a sobel orientation texture histogram on the top half and one on the bottom half of the image, all three with n (>0) buckets, where n would be taken as input from the user.
'c0' or "cooc-0": A co-occurence matrix histogram features (see the report for more detail about the exact features) with axis = 0, distance = 5.
'c1' or "cooc-1": A co-occurence matrix histogram features (see the report for more detail about the exact features) with axis = 1, distance = 5.
'c' or "cooc": A co-occurence matrix histogram features (see the report for more detail about the exact features) with axis = a (either 0 or 1), distance = d (>0, <=10), where a and d would be taken as input fromt he user.
'h-c0' or "histogram-cooc-0": A rg-chromaticity histogram (with 32 buckets) combined with co-occurence matrix histogram features (see the report for more detail about the exact features) with axis = 0, distance = 5.
'h-c1' or "histogram-cooc-1": A rg-chromaticity histogram (with 32 buckets) combined with co-occurence matrix histogram features (see the report for more detail about the exact features) with axis = 1, distance = 5.
'h-c' or "histogram-cooc": A rg-chromaticity histogram (with n (>0) buckets, taken as input) combined with co-occurence matrix histogram features (see the report for more detail about the exact features) with axis = a (either 0 or 1), distance = d (>0, <=10), where a and d would be taken as input fromt he user.
"pure-histogram": Uses a simple histogram created with just the pixel color values. The colors to be used to create the histogram would be taken as an input string (for example: 'rg' would result in a 2-d histogram with only the R and G color values of the pixel). The string should contain at least one of r, g, b.



Possible values of <D> and the distance metrics they correspond to:
"euclid" or "e" or "l2": euclidean distance or L-2 norm.
"l1": L-1 norm.
"ln": L-n norm, where n is taken as input from the user.
"hamming-distance" or "h": Hamming distance.
"histogram-intersection" or "i": Histogram intersection (actually a similarity metric and not a distance metric).

Note: by default the histograms will not be normalized before calculating these distances, to normalize append "-n" to the end of the distance metric argument strings (eg: e-n)

I've mentioned the exact commands to test the particular tasks in the report alogn with the respective results.