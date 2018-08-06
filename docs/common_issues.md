# Common issues

If your issue is not listed here, please search the [github issues](https://github.com/tensorflow/hub/issues) before filling a new one.


## Cannot download a module

In the process of using a module from an URL there are many errors that can show
up due to the network stack. Often this is a problem specific to the machine
running the code and not an issue with the library. Here is a list of the common
ones:

* **"EOF occured in violation of protocol"** - This issue is likely to be
generated if the installed python version does not support the TLS requirements
of the server hosting the module. Notably, python 2.7.5 is known to fail
resolving modules from tfhub.dev domain. **FIX**: Please update to a newer
python version.

* **"cannot verify tfhub.dev's certificate"** - This issue is likely to be
generated if something on the network is trying to act as the dev gTLD.
Before .dev was used as a gTLD, developers and frameworks would sometimes use
.dev names to help testing code. **FIX:** Identify and reconfigure the software
that intercepts name resolution in the ".dev" domain.

If the above errors and fixes do not work, one can try to manually download a
module by simulating the protocol of attaching `?tf-hub-format=compressed`
to the URL to download a tar compressed file that has to be manually decompressed
into a local file. The path to the local file can then be used instead of the
URL. Here is a quick example:

```bash
# Create a folder for the TF hub module.
$ mkdir /tmp/moduleA
# Download the module, and uncompress it to the destination folder. You might want to do this manually.
$ curl -L "https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed" | tar -zxvC /tmp/moduleA
# Test to make sure it works.
$ python
> import tensorflow_hub as hub
> hub.Module("/tmp/moduleA")
```
