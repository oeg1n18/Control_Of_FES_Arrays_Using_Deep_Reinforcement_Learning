χ
έ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018Έτ
¨
(Adam/actor_critic_network/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/actor_critic_network/dense_3/bias/v
‘
<Adam/actor_critic_network/dense_3/bias/v/Read/ReadVariableOpReadVariableOp(Adam/actor_critic_network/dense_3/bias/v*
_output_shapes
:*
dtype0
±
*Adam/actor_critic_network/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/actor_critic_network/dense_3/kernel/v
ͺ
>Adam/actor_critic_network/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/actor_critic_network/dense_3/kernel/v*
_output_shapes
:	*
dtype0
¨
(Adam/actor_critic_network/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/actor_critic_network/dense_2/bias/v
‘
<Adam/actor_critic_network/dense_2/bias/v/Read/ReadVariableOpReadVariableOp(Adam/actor_critic_network/dense_2/bias/v*
_output_shapes
:*
dtype0
±
*Adam/actor_critic_network/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/actor_critic_network/dense_2/kernel/v
ͺ
>Adam/actor_critic_network/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/actor_critic_network/dense_2/kernel/v*
_output_shapes
:	*
dtype0
©
(Adam/actor_critic_network/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/actor_critic_network/dense_1/bias/v
’
<Adam/actor_critic_network/dense_1/bias/v/Read/ReadVariableOpReadVariableOp(Adam/actor_critic_network/dense_1/bias/v*
_output_shapes	
:*
dtype0
²
*Adam/actor_critic_network/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/actor_critic_network/dense_1/kernel/v
«
>Adam/actor_critic_network/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/actor_critic_network/dense_1/kernel/v* 
_output_shapes
:
*
dtype0
₯
&Adam/actor_critic_network/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/actor_critic_network/dense/bias/v

:Adam/actor_critic_network/dense/bias/v/Read/ReadVariableOpReadVariableOp&Adam/actor_critic_network/dense/bias/v*
_output_shapes	
:*
dtype0
­
(Adam/actor_critic_network/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(Adam/actor_critic_network/dense/kernel/v
¦
<Adam/actor_critic_network/dense/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/actor_critic_network/dense/kernel/v*
_output_shapes
:	*
dtype0
¨
(Adam/actor_critic_network/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/actor_critic_network/dense_3/bias/m
‘
<Adam/actor_critic_network/dense_3/bias/m/Read/ReadVariableOpReadVariableOp(Adam/actor_critic_network/dense_3/bias/m*
_output_shapes
:*
dtype0
±
*Adam/actor_critic_network/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/actor_critic_network/dense_3/kernel/m
ͺ
>Adam/actor_critic_network/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/actor_critic_network/dense_3/kernel/m*
_output_shapes
:	*
dtype0
¨
(Adam/actor_critic_network/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/actor_critic_network/dense_2/bias/m
‘
<Adam/actor_critic_network/dense_2/bias/m/Read/ReadVariableOpReadVariableOp(Adam/actor_critic_network/dense_2/bias/m*
_output_shapes
:*
dtype0
±
*Adam/actor_critic_network/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/actor_critic_network/dense_2/kernel/m
ͺ
>Adam/actor_critic_network/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/actor_critic_network/dense_2/kernel/m*
_output_shapes
:	*
dtype0
©
(Adam/actor_critic_network/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/actor_critic_network/dense_1/bias/m
’
<Adam/actor_critic_network/dense_1/bias/m/Read/ReadVariableOpReadVariableOp(Adam/actor_critic_network/dense_1/bias/m*
_output_shapes	
:*
dtype0
²
*Adam/actor_critic_network/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/actor_critic_network/dense_1/kernel/m
«
>Adam/actor_critic_network/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/actor_critic_network/dense_1/kernel/m* 
_output_shapes
:
*
dtype0
₯
&Adam/actor_critic_network/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/actor_critic_network/dense/bias/m

:Adam/actor_critic_network/dense/bias/m/Read/ReadVariableOpReadVariableOp&Adam/actor_critic_network/dense/bias/m*
_output_shapes	
:*
dtype0
­
(Adam/actor_critic_network/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(Adam/actor_critic_network/dense/kernel/m
¦
<Adam/actor_critic_network/dense/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/actor_critic_network/dense/kernel/m*
_output_shapes
:	*
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

!actor_critic_network/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!actor_critic_network/dense_3/bias

5actor_critic_network/dense_3/bias/Read/ReadVariableOpReadVariableOp!actor_critic_network/dense_3/bias*
_output_shapes
:*
dtype0
£
#actor_critic_network/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#actor_critic_network/dense_3/kernel

7actor_critic_network/dense_3/kernel/Read/ReadVariableOpReadVariableOp#actor_critic_network/dense_3/kernel*
_output_shapes
:	*
dtype0

!actor_critic_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!actor_critic_network/dense_2/bias

5actor_critic_network/dense_2/bias/Read/ReadVariableOpReadVariableOp!actor_critic_network/dense_2/bias*
_output_shapes
:*
dtype0
£
#actor_critic_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#actor_critic_network/dense_2/kernel

7actor_critic_network/dense_2/kernel/Read/ReadVariableOpReadVariableOp#actor_critic_network/dense_2/kernel*
_output_shapes
:	*
dtype0

!actor_critic_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!actor_critic_network/dense_1/bias

5actor_critic_network/dense_1/bias/Read/ReadVariableOpReadVariableOp!actor_critic_network/dense_1/bias*
_output_shapes	
:*
dtype0
€
#actor_critic_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#actor_critic_network/dense_1/kernel

7actor_critic_network/dense_1/kernel/Read/ReadVariableOpReadVariableOp#actor_critic_network/dense_1/kernel* 
_output_shapes
:
*
dtype0

actor_critic_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!actor_critic_network/dense/bias

3actor_critic_network/dense/bias/Read/ReadVariableOpReadVariableOpactor_critic_network/dense/bias*
_output_shapes	
:*
dtype0

!actor_critic_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!actor_critic_network/dense/kernel

5actor_critic_network/dense/kernel/Read/ReadVariableOpReadVariableOp!actor_critic_network/dense/kernel*
_output_shapes
:	*
dtype0

NoOpNoOp
θ4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*£4
value4B4 B4
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2
	
value

policy
	optimizer
loss

signatures*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
¦
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

kernel
bias*
¦
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

kernel
bias*
¦
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

kernel
bias*
¦
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

kernel
bias*
Τ
8iter

9beta_1

:beta_2
	;decay
<learning_ratemZm[m\m]m^m_m`mavbvcvdvevfvgvhvi*
* 

=serving_default* 
a[
VARIABLE_VALUE!actor_critic_network/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEactor_critic_network/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#actor_critic_network/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!actor_critic_network/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#actor_critic_network/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!actor_critic_network/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#actor_critic_network/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!actor_critic_network/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 

0
1*

0
1*
* 

Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 

0
1*

0
1*
* 

Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

Qtrace_0* 

Rtrace_0* 

0
1*

0
1*
* 

Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
~
VARIABLE_VALUE(Adam/actor_critic_network/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/actor_critic_network/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/actor_critic_network/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/actor_critic_network/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/actor_critic_network/dense_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/actor_critic_network/dense_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/actor_critic_network/dense_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/actor_critic_network/dense_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/actor_critic_network/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/actor_critic_network/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/actor_critic_network/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/actor_critic_network/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/actor_critic_network/dense_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/actor_critic_network/dense_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/actor_critic_network/dense_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/actor_critic_network/dense_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
υ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!actor_critic_network/dense/kernelactor_critic_network/dense/bias#actor_critic_network/dense_1/kernel!actor_critic_network/dense_1/bias#actor_critic_network/dense_2/kernel!actor_critic_network/dense_2/bias#actor_critic_network/dense_3/kernel!actor_critic_network/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_137288404
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5actor_critic_network/dense/kernel/Read/ReadVariableOp3actor_critic_network/dense/bias/Read/ReadVariableOp7actor_critic_network/dense_1/kernel/Read/ReadVariableOp5actor_critic_network/dense_1/bias/Read/ReadVariableOp7actor_critic_network/dense_2/kernel/Read/ReadVariableOp5actor_critic_network/dense_2/bias/Read/ReadVariableOp7actor_critic_network/dense_3/kernel/Read/ReadVariableOp5actor_critic_network/dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp<Adam/actor_critic_network/dense/kernel/m/Read/ReadVariableOp:Adam/actor_critic_network/dense/bias/m/Read/ReadVariableOp>Adam/actor_critic_network/dense_1/kernel/m/Read/ReadVariableOp<Adam/actor_critic_network/dense_1/bias/m/Read/ReadVariableOp>Adam/actor_critic_network/dense_2/kernel/m/Read/ReadVariableOp<Adam/actor_critic_network/dense_2/bias/m/Read/ReadVariableOp>Adam/actor_critic_network/dense_3/kernel/m/Read/ReadVariableOp<Adam/actor_critic_network/dense_3/bias/m/Read/ReadVariableOp<Adam/actor_critic_network/dense/kernel/v/Read/ReadVariableOp:Adam/actor_critic_network/dense/bias/v/Read/ReadVariableOp>Adam/actor_critic_network/dense_1/kernel/v/Read/ReadVariableOp<Adam/actor_critic_network/dense_1/bias/v/Read/ReadVariableOp>Adam/actor_critic_network/dense_2/kernel/v/Read/ReadVariableOp<Adam/actor_critic_network/dense_2/bias/v/Read/ReadVariableOp>Adam/actor_critic_network/dense_3/kernel/v/Read/ReadVariableOp<Adam/actor_critic_network/dense_3/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_save_137288649
Ώ

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!actor_critic_network/dense/kernelactor_critic_network/dense/bias#actor_critic_network/dense_1/kernel!actor_critic_network/dense_1/bias#actor_critic_network/dense_2/kernel!actor_critic_network/dense_2/bias#actor_critic_network/dense_3/kernel!actor_critic_network/dense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate(Adam/actor_critic_network/dense/kernel/m&Adam/actor_critic_network/dense/bias/m*Adam/actor_critic_network/dense_1/kernel/m(Adam/actor_critic_network/dense_1/bias/m*Adam/actor_critic_network/dense_2/kernel/m(Adam/actor_critic_network/dense_2/bias/m*Adam/actor_critic_network/dense_3/kernel/m(Adam/actor_critic_network/dense_3/bias/m(Adam/actor_critic_network/dense/kernel/v&Adam/actor_critic_network/dense/bias/v*Adam/actor_critic_network/dense_1/kernel/v(Adam/actor_critic_network/dense_1/bias/v*Adam/actor_critic_network/dense_2/kernel/v(Adam/actor_critic_network/dense_2/bias/v*Adam/actor_critic_network/dense_3/kernel/v(Adam/actor_critic_network/dense_3/bias/v*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__traced_restore_137288746»Ψ
Ν

+__inference_dense_1_layer_call_fn_137288488

inputs
unknown:

	unknown_0:	
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_137288223p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
γE
ϋ
"__inference__traced_save_137288649
file_prefix@
<savev2_actor_critic_network_dense_kernel_read_readvariableop>
:savev2_actor_critic_network_dense_bias_read_readvariableopB
>savev2_actor_critic_network_dense_1_kernel_read_readvariableop@
<savev2_actor_critic_network_dense_1_bias_read_readvariableopB
>savev2_actor_critic_network_dense_2_kernel_read_readvariableop@
<savev2_actor_critic_network_dense_2_bias_read_readvariableopB
>savev2_actor_critic_network_dense_3_kernel_read_readvariableop@
<savev2_actor_critic_network_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopG
Csavev2_adam_actor_critic_network_dense_kernel_m_read_readvariableopE
Asavev2_adam_actor_critic_network_dense_bias_m_read_readvariableopI
Esavev2_adam_actor_critic_network_dense_1_kernel_m_read_readvariableopG
Csavev2_adam_actor_critic_network_dense_1_bias_m_read_readvariableopI
Esavev2_adam_actor_critic_network_dense_2_kernel_m_read_readvariableopG
Csavev2_adam_actor_critic_network_dense_2_bias_m_read_readvariableopI
Esavev2_adam_actor_critic_network_dense_3_kernel_m_read_readvariableopG
Csavev2_adam_actor_critic_network_dense_3_bias_m_read_readvariableopG
Csavev2_adam_actor_critic_network_dense_kernel_v_read_readvariableopE
Asavev2_adam_actor_critic_network_dense_bias_v_read_readvariableopI
Esavev2_adam_actor_critic_network_dense_1_kernel_v_read_readvariableopG
Csavev2_adam_actor_critic_network_dense_1_bias_v_read_readvariableopI
Esavev2_adam_actor_critic_network_dense_2_kernel_v_read_readvariableopG
Csavev2_adam_actor_critic_network_dense_2_bias_v_read_readvariableopI
Esavev2_adam_actor_critic_network_dense_3_kernel_v_read_readvariableopG
Csavev2_adam_actor_critic_network_dense_3_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ο
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH©
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ε
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_actor_critic_network_dense_kernel_read_readvariableop:savev2_actor_critic_network_dense_bias_read_readvariableop>savev2_actor_critic_network_dense_1_kernel_read_readvariableop<savev2_actor_critic_network_dense_1_bias_read_readvariableop>savev2_actor_critic_network_dense_2_kernel_read_readvariableop<savev2_actor_critic_network_dense_2_bias_read_readvariableop>savev2_actor_critic_network_dense_3_kernel_read_readvariableop<savev2_actor_critic_network_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopCsavev2_adam_actor_critic_network_dense_kernel_m_read_readvariableopAsavev2_adam_actor_critic_network_dense_bias_m_read_readvariableopEsavev2_adam_actor_critic_network_dense_1_kernel_m_read_readvariableopCsavev2_adam_actor_critic_network_dense_1_bias_m_read_readvariableopEsavev2_adam_actor_critic_network_dense_2_kernel_m_read_readvariableopCsavev2_adam_actor_critic_network_dense_2_bias_m_read_readvariableopEsavev2_adam_actor_critic_network_dense_3_kernel_m_read_readvariableopCsavev2_adam_actor_critic_network_dense_3_bias_m_read_readvariableopCsavev2_adam_actor_critic_network_dense_kernel_v_read_readvariableopAsavev2_adam_actor_critic_network_dense_bias_v_read_readvariableopEsavev2_adam_actor_critic_network_dense_1_kernel_v_read_readvariableopCsavev2_adam_actor_critic_network_dense_1_bias_v_read_readvariableopEsavev2_adam_actor_critic_network_dense_2_kernel_v_read_readvariableopCsavev2_adam_actor_critic_network_dense_2_bias_v_read_readvariableopEsavev2_adam_actor_critic_network_dense_3_kernel_v_read_readvariableopCsavev2_adam_actor_critic_network_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ψ
_input_shapesζ
γ: :	::
::	::	:: : : : : :	::
::	::	::	::
::	::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: 
©

ϊ
F__inference_dense_1_layer_call_and_return_conditional_losses_137288223

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
 
ί
8__inference_actor_critic_network_layer_call_fn_137288285
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
identity

identity_1’StatefulPartitionedCallΛ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_137288264o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
£

χ
D__inference_dense_layer_call_and_return_conditional_losses_137288479

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
©

ϊ
F__inference_dense_1_layer_call_and_return_conditional_losses_137288499

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ι

+__inference_dense_2_layer_call_fn_137288508

inputs
unknown:	
	unknown_0:
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_137288239o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ν	
ψ
F__inference_dense_2_layer_call_and_return_conditional_losses_137288518

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζ

)__inference_dense_layer_call_fn_137288468

inputs
unknown:	
	unknown_0:	
identity’StatefulPartitionedCallΪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_137288206p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Χz
Ζ
%__inference__traced_restore_137288746
file_prefixE
2assignvariableop_actor_critic_network_dense_kernel:	A
2assignvariableop_1_actor_critic_network_dense_bias:	J
6assignvariableop_2_actor_critic_network_dense_1_kernel:
C
4assignvariableop_3_actor_critic_network_dense_1_bias:	I
6assignvariableop_4_actor_critic_network_dense_2_kernel:	B
4assignvariableop_5_actor_critic_network_dense_2_bias:I
6assignvariableop_6_actor_critic_network_dense_3_kernel:	B
4assignvariableop_7_actor_critic_network_dense_3_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: O
<assignvariableop_13_adam_actor_critic_network_dense_kernel_m:	I
:assignvariableop_14_adam_actor_critic_network_dense_bias_m:	R
>assignvariableop_15_adam_actor_critic_network_dense_1_kernel_m:
K
<assignvariableop_16_adam_actor_critic_network_dense_1_bias_m:	Q
>assignvariableop_17_adam_actor_critic_network_dense_2_kernel_m:	J
<assignvariableop_18_adam_actor_critic_network_dense_2_bias_m:Q
>assignvariableop_19_adam_actor_critic_network_dense_3_kernel_m:	J
<assignvariableop_20_adam_actor_critic_network_dense_3_bias_m:O
<assignvariableop_21_adam_actor_critic_network_dense_kernel_v:	I
:assignvariableop_22_adam_actor_critic_network_dense_bias_v:	R
>assignvariableop_23_adam_actor_critic_network_dense_1_kernel_v:
K
<assignvariableop_24_adam_actor_critic_network_dense_1_bias_v:	Q
>assignvariableop_25_adam_actor_critic_network_dense_2_kernel_v:	J
<assignvariableop_26_adam_actor_critic_network_dense_2_bias_v:Q
>assignvariableop_27_adam_actor_critic_network_dense_3_kernel_v:	J
<assignvariableop_28_adam_actor_critic_network_dense_3_bias_v:
identity_30’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9ς
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ΅
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp2assignvariableop_actor_critic_network_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:‘
AssignVariableOp_1AssignVariableOp2assignvariableop_1_actor_critic_network_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_2AssignVariableOp6assignvariableop_2_actor_critic_network_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_3AssignVariableOp4assignvariableop_3_actor_critic_network_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_4AssignVariableOp6assignvariableop_4_actor_critic_network_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_5AssignVariableOp4assignvariableop_5_actor_critic_network_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_6AssignVariableOp6assignvariableop_6_actor_critic_network_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_7AssignVariableOp4assignvariableop_7_actor_critic_network_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_13AssignVariableOp<assignvariableop_13_adam_actor_critic_network_dense_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_14AssignVariableOp:assignvariableop_14_adam_actor_critic_network_dense_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_15AssignVariableOp>assignvariableop_15_adam_actor_critic_network_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_16AssignVariableOp<assignvariableop_16_adam_actor_critic_network_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_17AssignVariableOp>assignvariableop_17_adam_actor_critic_network_dense_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_18AssignVariableOp<assignvariableop_18_adam_actor_critic_network_dense_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_19AssignVariableOp>assignvariableop_19_adam_actor_critic_network_dense_3_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_20AssignVariableOp<assignvariableop_20_adam_actor_critic_network_dense_3_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_21AssignVariableOp<assignvariableop_21_adam_actor_critic_network_dense_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_22AssignVariableOp:assignvariableop_22_adam_actor_critic_network_dense_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_actor_critic_network_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_24AssignVariableOp<assignvariableop_24_adam_actor_critic_network_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_25AssignVariableOp>assignvariableop_25_adam_actor_critic_network_dense_2_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_26AssignVariableOp<assignvariableop_26_adam_actor_critic_network_dense_2_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_actor_critic_network_dense_3_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_28AssignVariableOp<assignvariableop_28_adam_actor_critic_network_dense_3_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ν
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_30IdentityIdentity_29:output:0^NoOp_1*
T0*
_output_shapes
: Ί
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_30Identity_30:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ΰ

Ξ
'__inference_signature_wrapper_137288404
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
identity

identity_1’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__wrapped_model_137288188o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
$
Θ
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_137288459	
state7
$dense_matmul_readvariableop_resource:	4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	9
&dense_2_matmul_readvariableop_resource:	5
'dense_2_biasadd_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	5
'dense_3_biasadd_readvariableop_resource:
identity

identity_1’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’dense_2/BiasAdd/ReadVariableOp’dense_2/MatMul/ReadVariableOp’dense_3/BiasAdd/ReadVariableOp’dense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0u
dense/MatMulMatMulstate#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_3/MatMulMatMuldense_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????i

Identity_1Identitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Ζ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_namestate
Ι

+__inference_dense_3_layer_call_fn_137288527

inputs
unknown:	
	unknown_0:
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_137288256o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

έ
8__inference_actor_critic_network_layer_call_fn_137288427	
state
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
identity

identity_1’StatefulPartitionedCallΙ
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_137288264o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namestate
₯
’
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_137288264	
state"
dense_137288207:	
dense_137288209:	%
dense_1_137288224:
 
dense_1_137288226:	$
dense_2_137288240:	
dense_2_137288242:$
dense_3_137288257:	
dense_3_137288259:
identity

identity_1’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCall’dense_3/StatefulPartitionedCallν
dense/StatefulPartitionedCallStatefulPartitionedCallstatedense_137288207dense_137288209*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_137288206
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_137288224dense_1_137288226*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_137288223
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_137288240dense_2_137288242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_137288239
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_137288257dense_3_137288259*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_137288256w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y

Identity_1Identity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Μ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namestate
¦

ψ
F__inference_dense_3_layer_call_and_return_conditional_losses_137288256

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
«
€
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_137288373
input_1"
dense_137288351:	
dense_137288353:	%
dense_1_137288356:
 
dense_1_137288358:	$
dense_2_137288361:	
dense_2_137288363:$
dense_3_137288366:	
dense_3_137288368:
identity

identity_1’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCall’dense_3/StatefulPartitionedCallο
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_137288351dense_137288353*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_137288206
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_137288356dense_1_137288358*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_137288223
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_137288361dense_2_137288363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_137288239
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_137288366dense_3_137288368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_137288256w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y

Identity_1Identity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Μ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
2
λ
$__inference__wrapped_model_137288188
input_1L
9actor_critic_network_dense_matmul_readvariableop_resource:	I
:actor_critic_network_dense_biasadd_readvariableop_resource:	O
;actor_critic_network_dense_1_matmul_readvariableop_resource:
K
<actor_critic_network_dense_1_biasadd_readvariableop_resource:	N
;actor_critic_network_dense_2_matmul_readvariableop_resource:	J
<actor_critic_network_dense_2_biasadd_readvariableop_resource:N
;actor_critic_network_dense_3_matmul_readvariableop_resource:	J
<actor_critic_network_dense_3_biasadd_readvariableop_resource:
identity

identity_1’1actor_critic_network/dense/BiasAdd/ReadVariableOp’0actor_critic_network/dense/MatMul/ReadVariableOp’3actor_critic_network/dense_1/BiasAdd/ReadVariableOp’2actor_critic_network/dense_1/MatMul/ReadVariableOp’3actor_critic_network/dense_2/BiasAdd/ReadVariableOp’2actor_critic_network/dense_2/MatMul/ReadVariableOp’3actor_critic_network/dense_3/BiasAdd/ReadVariableOp’2actor_critic_network/dense_3/MatMul/ReadVariableOp«
0actor_critic_network/dense/MatMul/ReadVariableOpReadVariableOp9actor_critic_network_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0‘
!actor_critic_network/dense/MatMulMatMulinput_18actor_critic_network/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????©
1actor_critic_network/dense/BiasAdd/ReadVariableOpReadVariableOp:actor_critic_network_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Θ
"actor_critic_network/dense/BiasAddBiasAdd+actor_critic_network/dense/MatMul:product:09actor_critic_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
actor_critic_network/dense/ReluRelu+actor_critic_network/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????°
2actor_critic_network/dense_1/MatMul/ReadVariableOpReadVariableOp;actor_critic_network_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Λ
#actor_critic_network/dense_1/MatMulMatMul-actor_critic_network/dense/Relu:activations:0:actor_critic_network/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????­
3actor_critic_network/dense_1/BiasAdd/ReadVariableOpReadVariableOp<actor_critic_network_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ξ
$actor_critic_network/dense_1/BiasAddBiasAdd-actor_critic_network/dense_1/MatMul:product:0;actor_critic_network/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
!actor_critic_network/dense_1/ReluRelu-actor_critic_network/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????―
2actor_critic_network/dense_2/MatMul/ReadVariableOpReadVariableOp;actor_critic_network_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Μ
#actor_critic_network/dense_2/MatMulMatMul/actor_critic_network/dense_1/Relu:activations:0:actor_critic_network/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????¬
3actor_critic_network/dense_2/BiasAdd/ReadVariableOpReadVariableOp<actor_critic_network_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ν
$actor_critic_network/dense_2/BiasAddBiasAdd-actor_critic_network/dense_2/MatMul:product:0;actor_critic_network/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????―
2actor_critic_network/dense_3/MatMul/ReadVariableOpReadVariableOp;actor_critic_network_dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Μ
#actor_critic_network/dense_3/MatMulMatMul/actor_critic_network/dense_1/Relu:activations:0:actor_critic_network/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????¬
3actor_critic_network/dense_3/BiasAdd/ReadVariableOpReadVariableOp<actor_critic_network_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ν
$actor_critic_network/dense_3/BiasAddBiasAdd-actor_critic_network/dense_3/MatMul:product:0;actor_critic_network/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$actor_critic_network/dense_3/SoftmaxSoftmax-actor_critic_network/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
IdentityIdentity.actor_critic_network/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????~

Identity_1Identity-actor_critic_network/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????ξ
NoOpNoOp2^actor_critic_network/dense/BiasAdd/ReadVariableOp1^actor_critic_network/dense/MatMul/ReadVariableOp4^actor_critic_network/dense_1/BiasAdd/ReadVariableOp3^actor_critic_network/dense_1/MatMul/ReadVariableOp4^actor_critic_network/dense_2/BiasAdd/ReadVariableOp3^actor_critic_network/dense_2/MatMul/ReadVariableOp4^actor_critic_network/dense_3/BiasAdd/ReadVariableOp3^actor_critic_network/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2f
1actor_critic_network/dense/BiasAdd/ReadVariableOp1actor_critic_network/dense/BiasAdd/ReadVariableOp2d
0actor_critic_network/dense/MatMul/ReadVariableOp0actor_critic_network/dense/MatMul/ReadVariableOp2j
3actor_critic_network/dense_1/BiasAdd/ReadVariableOp3actor_critic_network/dense_1/BiasAdd/ReadVariableOp2h
2actor_critic_network/dense_1/MatMul/ReadVariableOp2actor_critic_network/dense_1/MatMul/ReadVariableOp2j
3actor_critic_network/dense_2/BiasAdd/ReadVariableOp3actor_critic_network/dense_2/BiasAdd/ReadVariableOp2h
2actor_critic_network/dense_2/MatMul/ReadVariableOp2actor_critic_network/dense_2/MatMul/ReadVariableOp2j
3actor_critic_network/dense_3/BiasAdd/ReadVariableOp3actor_critic_network/dense_3/BiasAdd/ReadVariableOp2h
2actor_critic_network/dense_3/MatMul/ReadVariableOp2actor_critic_network/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
Ν	
ψ
F__inference_dense_2_layer_call_and_return_conditional_losses_137288239

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
£

χ
D__inference_dense_layer_call_and_return_conditional_losses_137288206

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
¦

ψ
F__inference_dense_3_layer_call_and_return_conditional_losses_137288538

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs"ΏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ι
serving_defaultΥ
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????<
output_20
StatefulPartitionedCall:1?????????tensorflow/serving/predict:΄q

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2
	
value

policy
	optimizer
loss

signatures"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ο
trace_0
trace_12
8__inference_actor_critic_network_layer_call_fn_137288285
8__inference_actor_critic_network_layer_call_fn_137288427‘
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0ztrace_1

trace_0
trace_12Ξ
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_137288459
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_137288373‘
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0ztrace_1
ΟBΜ
$__inference__wrapped_model_137288188input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
»
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
γ
8iter

9beta_1

:beta_2
	;decay
<learning_ratemZm[m\m]m^m_m`mavbvcvdvevfvgvhvi"
	optimizer
 "
trackable_dict_wrapper
,
=serving_default"
signature_map
4:2	2!actor_critic_network/dense/kernel
.:,2actor_critic_network/dense/bias
7:5
2#actor_critic_network/dense_1/kernel
0:.2!actor_critic_network/dense_1/bias
6:4	2#actor_critic_network/dense_2/kernel
/:-2!actor_critic_network/dense_2/bias
6:4	2#actor_critic_network/dense_3/kernel
/:-2!actor_critic_network/dense_3/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
μBι
8__inference_actor_critic_network_layer_call_fn_137288285input_1"‘
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
κBη
8__inference_actor_critic_network_layer_call_fn_137288427state"‘
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_137288459state"‘
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_137288373input_1"‘
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ν
Ctrace_02Π
)__inference_dense_layer_call_fn_137288468’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zCtrace_0

Dtrace_02λ
D__inference_dense_layer_call_and_return_conditional_losses_137288479’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zDtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
ο
Jtrace_02?
+__inference_dense_1_layer_call_fn_137288488’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zJtrace_0

Ktrace_02ν
F__inference_dense_1_layer_call_and_return_conditional_losses_137288499’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zKtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
ο
Qtrace_02?
+__inference_dense_2_layer_call_fn_137288508’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zQtrace_0

Rtrace_02ν
F__inference_dense_2_layer_call_and_return_conditional_losses_137288518’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zRtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
ο
Xtrace_02?
+__inference_dense_3_layer_call_fn_137288527’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zXtrace_0

Ytrace_02ν
F__inference_dense_3_layer_call_and_return_conditional_losses_137288538’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zYtrace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ΞBΛ
'__inference_signature_wrapper_137288404input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_dense_layer_call_fn_137288468inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_dense_layer_call_and_return_conditional_losses_137288479inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ίBά
+__inference_dense_1_layer_call_fn_137288488inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊBχ
F__inference_dense_1_layer_call_and_return_conditional_losses_137288499inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ίBά
+__inference_dense_2_layer_call_fn_137288508inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊBχ
F__inference_dense_2_layer_call_and_return_conditional_losses_137288518inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ίBά
+__inference_dense_3_layer_call_fn_137288527inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊBχ
F__inference_dense_3_layer_call_and_return_conditional_losses_137288538inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
9:7	2(Adam/actor_critic_network/dense/kernel/m
3:12&Adam/actor_critic_network/dense/bias/m
<::
2*Adam/actor_critic_network/dense_1/kernel/m
5:32(Adam/actor_critic_network/dense_1/bias/m
;:9	2*Adam/actor_critic_network/dense_2/kernel/m
4:22(Adam/actor_critic_network/dense_2/bias/m
;:9	2*Adam/actor_critic_network/dense_3/kernel/m
4:22(Adam/actor_critic_network/dense_3/bias/m
9:7	2(Adam/actor_critic_network/dense/kernel/v
3:12&Adam/actor_critic_network/dense/bias/v
<::
2*Adam/actor_critic_network/dense_1/kernel/v
5:32(Adam/actor_critic_network/dense_1/bias/v
;:9	2*Adam/actor_critic_network/dense_2/kernel/v
4:22(Adam/actor_critic_network/dense_2/bias/v
;:9	2*Adam/actor_critic_network/dense_3/kernel/v
4:22(Adam/actor_critic_network/dense_3/bias/vΚ
$__inference__wrapped_model_137288188‘0’-
&’#
!
input_1?????????
ͺ "cͺ`
.
output_1"
output_1?????????
.
output_2"
output_2?????????α
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_1372883730’-
&’#
!
input_1?????????
ͺ "K’H
A’>

0/0?????????

0/1?????????
 ί
S__inference_actor_critic_network_layer_call_and_return_conditional_losses_137288459.’+
$’!

state?????????
ͺ "K’H
A’>

0/0?????????

0/1?????????
 ·
8__inference_actor_critic_network_layer_call_fn_137288285{0’-
&’#
!
input_1?????????
ͺ "=’:

0?????????

1?????????΅
8__inference_actor_critic_network_layer_call_fn_137288427y.’+
$’!

state?????????
ͺ "=’:

0?????????

1?????????¨
F__inference_dense_1_layer_call_and_return_conditional_losses_137288499^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
+__inference_dense_1_layer_call_fn_137288488Q0’-
&’#
!
inputs?????????
ͺ "?????????§
F__inference_dense_2_layer_call_and_return_conditional_losses_137288518]0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????
 
+__inference_dense_2_layer_call_fn_137288508P0’-
&’#
!
inputs?????????
ͺ "?????????§
F__inference_dense_3_layer_call_and_return_conditional_losses_137288538]0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????
 
+__inference_dense_3_layer_call_fn_137288527P0’-
&’#
!
inputs?????????
ͺ "?????????₯
D__inference_dense_layer_call_and_return_conditional_losses_137288479]/’,
%’"
 
inputs?????????
ͺ "&’#

0?????????
 }
)__inference_dense_layer_call_fn_137288468P/’,
%’"
 
inputs?????????
ͺ "?????????Ψ
'__inference_signature_wrapper_137288404¬;’8
’ 
1ͺ.
,
input_1!
input_1?????????"cͺ`
.
output_1"
output_1?????????
.
output_2"
output_2?????????