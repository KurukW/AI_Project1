$	j�t��?��q�]�?a2U0*�s?!j�q����?	|}~�%�?Y%��=	@!�6�%@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$j�q����?x$(~�?A�3��7��?Y�Y��ڊ�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails_�Qڋ?-C��6J?A-C��6�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�C�l���?��_�LU?A�5�;N��?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��ׁsF�?lxz�,C|?A�~j�t�h?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�&S��?-C��6Z?A���Q�~?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsHP�sׂ?_�Q�{?Aa2U0*�c?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�+e�X�?����Mb�?A_�Q�k?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsa2U0*�s?y�&1�l?A��_�LU?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�HP�x?{�G�zt?A/n��R?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	lxz�,C|?HP�s�r?AHP�s�b?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
M�O��?/n���?A��_�LU?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails{�G�zt?ŏ1w-!o?Aa2U0*�S?*	�����C@2F
Iterator::ModelHP�sע?!	Z���%X@)O��e�c�?1�O�S��R@:Preprocessing2U
Iterator::Model::ParallelMapV2��ǘ���?!Q(
�B5@)��ǘ���?1Q(
�B5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��_�LU?!Ӿ�/�K@)��_�LU?1Ӿ�/�K@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t21.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�����@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	[��@]��?����?��?-C��6J?!x$(~�?	!       "	!       *	!       2$	�?�ͦ?
���;@�?/n��R?!�3��7��?:	!       B	!       J	#��&�s?Q,��j�?!�Y��ڊ�?R	!       Z	#��&�s?Q,��j�?!�Y��ڊ�?JCPU_ONLYY�����@b 