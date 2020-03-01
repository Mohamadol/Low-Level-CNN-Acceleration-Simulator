#ifndef POOLING_TEST_H 
#define POOLING_TEST_H

#include"pooling.h"


double input[300] = {
1.624345363663241670e+00,-6.117564136500753813e-01,-5.281717522634556961e-01,-1.072968622156170504e+00,8.654076293246785179e-01,-2.301538696880282675e+00,1.744811764216479988e+00,-7.612069008951027893e-01,3.190390960570985701e-01,-2.493703754774100889e-01,1.462107937044974104e+00,-2.060140709497654044e+00,-3.224172040135074857e-01,-3.840543546684156428e-01,1.133769442335437416e+00,-1.099891267314030863e+00,-1.724282075504357525e-01,-8.778584179213717587e-01,4.221374671559283143e-02,5.828152137158222335e-01,-1.100619177212921240e+00,1.144723709839614134e+00,9.015907205927955470e-01,5.024943389018682316e-01,9.008559492644118150e-01,-6.837278591743330969e-01,-1.228902255186481718e-01,-9.357694342590687775e-01,-2.678880796260159070e-01,5.303554667381860099e-01,-6.916607517253090531e-01,-3.967535268559773676e-01,-6.871727001195994111e-01,-8.452056414987195732e-01,-6.712461308368190549e-01,-1.266459891890136039e-02,-1.117310348635277784e+00,2.344156978170921501e-01,1.659802177109870547e+00,7.420441605773355764e-01,-1.918355523616149250e-01,-8.876289640848362694e-01,-7.471582937508376432e-01,1.692454601027746586e+00,5.080775477602896689e-02,-6.369956465693533687e-01,1.909154846674660233e-01,2.100255136478842211e+00,1.201589524816291527e-01,6.172031097074192063e-01,3.001703199558274915e-01,-3.522498464935186480e-01,-1.142518198022140163e+00,-3.493427224128775044e-01,-2.088942333747781099e-01,5.866231911821976341e-01,8.389834138745049241e-01,9.311020813035573163e-01,2.855873252542587837e-01,8.851411642707280869e-01,-7.543979409966528049e-01,1.252868155233287872e+00,5.129298204180088305e-01,-2.980928351027156697e-01,4.885181465374970333e-01,-7.557171302105572530e-02,1.131629387451426938e+00,1.519816816422198791e+00,2.185575406533161402e+00,-1.396496335488137674e+00,-1.444113805429589448e+00,-5.044658629464512378e-01,1.600370694478304712e-01,8.761689211162249080e-01,3.156349472416052349e-01,-2.022201215824003029e+00,-3.062040126283718244e-01,8.279746426072461807e-01,2.300947353643834070e-01,7.620111803120247496e-01,-2.223281426103592695e-01,-2.007580689299974519e-01,1.865613909882843091e-01,4.100516472082563202e-01,1.982997201267697462e-01,1.190086458074588183e-01,-6.706622862890305736e-01,3.775637863209194145e-01,1.218212709914369279e-01,1.129483907911919660e+00,1.198917879901507000e+00,1.851564174839438470e-01,-3.752849500901141999e-01,-6.387304074542223820e-01,4.234943540641128989e-01,7.734006834855941537e-02,-3.438536755710756010e-01,4.359685683424693869e-02,-6.200008439481292655e-01,6.980320340722189210e-01,-4.471285647859982171e-01,1.224507704805498864e+00,4.034916417907999930e-01,5.935785232370669462e-01,-1.094911845741041834e+00,1.693824330586680971e-01,7.405564510962747704e-01,-9.537006018079345893e-01,-2.662185060036220685e-01,3.261454669335855927e-02,-1.373117320246755702e+00,3.151593920422917638e-01,8.461606475850333986e-01,-8.595159408319863470e-01,3.505459786641073605e-01,-1.312283411237431752e+00,-3.869550926605111463e-02,-1.615772354703294722e+00,1.121417708235664001e+00,4.089005379368277904e-01,-2.461695587577835548e-02,-7.751616191691595992e-01,1.273755930158776639e+00,1.967101749254734688e+00,-1.857981864446752063e+00,1.236164030452820306e+00,1.627650753148906393e+00,3.380116965744757729e-01,-1.199268032335186085e+00,8.633453175440215510e-01,-1.809203020781504634e-01,-6.039206277932572808e-01,-1.230058135666961761e+00,5.505374959762153741e-01,7.928068659193476808e-01,-6.235307296797916177e-01,5.205763370733708095e-01,-1.144341389623142691e+00,8.018610318713447205e-01,4.656729842414553816e-02,-1.865697719073487748e-01,-1.017458725291452148e-01,8.688861570058679096e-01,7.504116398650081399e-01,5.294653243527092101e-01,1.377012099973860815e-01,7.782112791270591468e-02,6.183802619985244720e-01,2.324945591787378751e-01,6.825514068644851218e-01,-3.101167735180599960e-01,-2.434837764107138813e+00,1.038824601859414054e+00,2.186979646974257729e+00,4.413644435685820655e-01,-1.001552332834997755e-01,-1.364447438960330328e-01,-1.190541877748098887e-01,1.740940830000459877e-02,-1.122018728746888350e+00,-5.170944579202279012e-01,-9.970268276502629590e-01,2.487991613877705011e-01,-2.966411523708627485e-01,4.952113239779604159e-01,-1.747031597425009464e-01,9.863351878212419654e-01,2.135339013354417836e-01,2.190699728969733417e+00,-1.896360922891092482e+00,-6.469166882549080011e-01,9.014868916487109862e-01,2.528325706806398010e+00,-2.486347777154600536e-01,4.366899317838910527e-02,-2.263142425136051850e-01,1.331457112587591807e+00,-2.873078634760188876e-01,6.800698398781045428e-01,-3.198015988986712133e-01,-1.272558755245994266e+00,3.135477204634321557e-01,5.031848134353260615e-01,1.293225882532261783e+00,-1.104470264173163102e-01,-6.173620637123609090e-01,5.627610966190262909e-01,2.407370922377322364e-01,2.806650771226390506e-01,-7.311270374727776855e-02,1.160338569993769609e+00,3.694927163757237287e-01,1.904658708340981166e+00,1.111056698560504596e+00,6.590497961002101945e-01,-1.627438340616257362e+00,6.023192802956289782e-01,4.202822036470595379e-01,8.109516728035557342e-01,1.044442094707258795e+00,-4.008781917889266411e-01,8.240056184504076509e-01,-5.623054310190898075e-01,1.954878075009034433e+00,-1.331951666517248167e+00,-1.760688560398783409e+00,-1.650721265824100170e+00,-8.905555841630484748e-01,-1.119115398559727970e+00,1.956078903703641902e+00,-3.264994980781842360e-01,-1.342675789377435924e+00,1.114382976779791923e+00,-5.865239388215924832e-01,-1.236853376541397376e+00,8.758389276492994924e-01,6.233621765780327229e-01,-4.349566829552277136e-01,1.407540000241228606e+00,1.291015797107254448e-01,1.616949598857300163e+00,5.027408819999042988e-01,1.558805540619859320e+00,1.094026964254281725e-01,-1.219744396979032697e+00,2.449368649061397285e+00,-5.457741679825677261e-01,-1.988378628888967381e-01,-7.003985049212546610e-01,-2.033944489645584386e-01,2.426694410817945846e-01,2.018301788740040348e-01,6.610202875986929127e-01,1.792158208975566991e+00,-1.204645717885074463e-01,-1.233120735446426641e+00,-1.182318126509633638e+00,-6.657545181991265659e-01,-1.674195807618932053e+00,8.250298244389858704e-01,-4.982135636310781046e-01,-3.109849783028508785e-01,-1.891482838003701473e-03,-1.396620424595431897e+00,-8.613163607760420115e-01,6.747115256879723244e-01,6.185391307862931898e-01,-4.431719307006377617e-01,1.810534914125456307e+00,-1.305726922557737479e+00,-3.449872101549794623e-01,-2.308397431354694551e-01,-2.793085000146540153e+00,1.937528813616079759e+00,3.663320145400582595e-01,-1.044589381907791603e+00,2.051173442857444407e+00,5.856620001723824576e-01,4.295261400219644865e-01,-6.069983982000460854e-01,1.062227240352178054e-01,-1.525680316229357736e+00,7.950260944248447315e-01,-3.744383188432206522e-01,1.340481965546231335e-01,1.202054862199705809e+00,2.847481108490579338e-01,2.624674454632686116e-01,2.764993048221836558e-01,-7.332716038953128734e-01,8.360047194342687948e-01,1.543359110804483736e+00,7.588056600979309341e-01,8.849088144648833421e-01,-8.772815189181882856e-01,-8.677872228729256454e-01,-1.440876024291839919e+00,1.232253070828436048e+00,-2.541798676073683261e-01,1.399843942480985870e+00,-7.819116826868006687e-01,-4.375089828285809723e-01,9.542508719125769590e-02,9.214500686595114010e-01,6.075019579950674542e-02,2.111247550077167412e-01,1.652756730561560916e-02,1.771877202759604142e-01,-1.116470017884744426e+00,8.092710097327859842e-02,-1.865789935114662845e-01,-5.682448088584730189e-02,4.923365559366488231e-01,-6.806781410088857953e-01,-8.450802740462980134e-02,-2.973618827735036163e-01,4.173020049748625282e-01,7.847706510155895154e-01,-9.554252623736891881e-01,5.859104311026156475e-01
};



void pooling_test();

void pooling_test() {
	int c_out, h_out, w_out;
	int in_size[3] = { 10,10,3 };
	pooling *layer = pooling_init(in_size, input);
	pooling_process(layer);


	for (c_out = 0; c_out < layer->output_c; c_out++) {
		printf("\nLayer");
		for (h_out = 0; h_out < layer->output_h; h_out++) {
			printf("\n");
			for (w_out = 0; w_out < layer->output_w; w_out++) {
				printf(" ** %.4f", *(layer->output + c_out * layer->output_h * layer->output_w + h_out * layer->output_w + w_out));
			}
		}
		printf("\nLayer end\n");
	}
}
#endif