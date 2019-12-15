#  A Software Implementation of Posit Arithmetic

*Antoine Sébert s193508 - 24/10/2019*

## Keywords

Floating-point Arithmetic, number format

## Abstract

This recently emerged floating-point number format aims to provide a better alternative to IEEE 754 arithmetic[^1]. It describes new data types called “posit” (float number) and “valid” (range of floats), formely know as unums of type I, II, and III[^2][^3]. The author expects this new format to produce higher accuracy and greater dynamic range without size overcost.

Although it is not exempt of critiques[^4], it seems a substitute interesting enough to be investigated[^5].

We propose to investigate this arithmetic, giving an overview of its advantages and downsides, and write a software implementation for demonstration and research purposes.

## Related Works

Some implementations, hardware or software, have already appeared, sustained by a community of researchers. They are hosted on a platform that also provide documentation and events notifications, to learn and discuss the topic[^6].

## References

[^1]: IEEE Standard for Floating-Point Arithmetic," in *IEEE Std 754-2019 (Revision of IEEE 754-2008)* , vol., no., pp.1-84, 22 July 2019 doi: 10.1109/IEEESTD.2019.8766229
[^2]: GUSTAFSON, John L. et YONEMOTO, Isaac T. Beating floating point at its own game: Posit arithmetic. *Supercomputing Frontiers and Innovations*, 2017, vol. 4, no 2, p. 71-86.
[^3]: GUSTAFSON, John L. Posit Arithmetic. *Online: https://posithub.org/docs/Posits4. pdf*, 2017.
[^4]: KAHAN, William. A critique of John L.  Gustafson’s The End of Error–Unum computation and his radical approach  to computation with real numbers. In : *23rd IEEE Symposium on Computer Arithmetic*. 2016.
[^5]: JAISWAL, Manish Kumar et SO, Hayden K.-H. Universal number posit arithmetic generator on FPGA. In : *2018 Design, Automation & Test in Europe Conference & Exhibition (DATE)*. IEEE, 2018. p. 1159-1162.
[^6]: https://posithub.org/docs/PDS/PositEffortsSurvey.html, retrieved 24/10/2019