Search.setIndex({docnames:["changes","index","m3gnet","m3gnet.callbacks","m3gnet.cli","m3gnet.config","m3gnet.graph","m3gnet.layers","m3gnet.models","m3gnet.trainers","m3gnet.type","m3gnet.utils","m3gnet.utils.constants","modules"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["changes.md","index.md","m3gnet.rst","m3gnet.callbacks.rst","m3gnet.cli.rst","m3gnet.config.rst","m3gnet.graph.rst","m3gnet.layers.rst","m3gnet.models.rst","m3gnet.trainers.rst","m3gnet.type.rst","m3gnet.utils.rst","m3gnet.utils.constants.rst","modules.rst"],objects:{"":{m3gnet:[2,0,0,"-"]},"m3gnet.callbacks":{ManualStop:[3,1,1,""]},"m3gnet.callbacks.ManualStop":{on_batch_end:[3,2,1,""]},"m3gnet.cli":{main:[4,3,1,""],relax_structure:[4,3,1,""]},"m3gnet.config":{DataType:[5,1,1,""],set_global_dtypes:[5,3,1,""]},"m3gnet.config.DataType":{np_float:[5,4,1,""],np_int:[5,4,1,""],set_dtype:[5,2,1,""],tf_float:[5,4,1,""],tf_int:[5,4,1,""]},"m3gnet.graph":{BaseGraphConverter:[6,1,1,""],Index:[6,1,1,""],MaterialGraph:[6,1,1,""],MaterialGraphBatch:[6,1,1,""],MaterialGraphBatchEnergyForceStress:[6,1,1,""],RadiusCutoffGraphConverter:[6,1,1,""],assemble_material_graph:[6,3,1,""],get_fixed_radius_bonding:[6,3,1,""],get_pair_vector_from_graph:[6,3,1,""],include_threebody_indices:[6,3,1,""],tf_compute_distance_angle:[6,3,1,""]},"m3gnet.graph.BaseGraphConverter":{convert:[6,2,1,""],convert_many:[6,2,1,""],get_atom_features:[6,2,1,""],get_states:[6,2,1,""],set_default_states:[6,2,1,""]},"m3gnet.graph.Index":{ATOMS:[6,4,1,""],ATOM_POSITIONS:[6,4,1,""],BONDS:[6,4,1,""],BOND_ATOM_INDICES:[6,4,1,""],BOND_WEIGHTS:[6,4,1,""],LATTICES:[6,4,1,""],N_ATOMS:[6,4,1,""],N_BONDS:[6,4,1,""],N_TRIPLE_I:[6,4,1,""],N_TRIPLE_IJ:[6,4,1,""],N_TRIPLE_S:[6,4,1,""],PBC_OFFSETS:[6,4,1,""],PHI:[6,4,1,""],STATES:[6,4,1,""],THETA:[6,4,1,""],TRIPLE_BOND_INDICES:[6,4,1,""],TRIPLE_BOND_LENGTHS:[6,4,1,""]},"m3gnet.graph.MaterialGraph":{as_list:[6,2,1,""],as_tf:[6,2,1,""],atom_positions:[6,4,1,""],atoms:[6,4,1,""],base_attributes:[6,4,1,""],bond_atom_indices:[6,4,1,""],bond_weights:[6,4,1,""],bonds:[6,4,1,""],copy:[6,2,1,""],from_list:[6,2,1,""],graph_with_position_attributes:[6,4,1,""],has_threebody:[6,5,1,""],lattices:[6,4,1,""],n_atom:[6,5,1,""],n_atoms:[6,4,1,""],n_bond:[6,5,1,""],n_bonds:[6,4,1,""],n_struct:[6,5,1,""],n_triple_i:[6,4,1,""],n_triple_ij:[6,4,1,""],n_triple_s:[6,4,1,""],pbc_offsets:[6,4,1,""],phi:[6,4,1,""],states:[6,4,1,""],theta:[6,4,1,""],three_body_attributes:[6,4,1,""],triple_bond_indices:[6,4,1,""],triple_bond_lengths:[6,4,1,""]},"m3gnet.graph.MaterialGraphBatch":{on_epoch_end:[6,2,1,""]},"m3gnet.graph.RadiusCutoffGraphConverter":{convert:[6,2,1,""],get_config:[6,2,1,""]},"m3gnet.layers":{AtomNetwork:[7,1,1,""],AtomReduceState:[7,1,1,""],AtomRef:[7,1,1,""],BaseAtomRef:[7,1,1,""],BondNetwork:[7,1,1,""],ConcatAtoms:[7,1,1,""],ConcatBondAtomState:[7,1,1,""],Embedding:[7,1,1,""],GatedAtomUpdate:[7,1,1,""],GatedMLP:[7,1,1,""],GraphFeaturizer:[7,1,1,""],GraphFieldEmbedding:[7,1,1,""],GraphNetworkLayer:[7,1,1,""],GraphUpdate:[7,1,1,""],GraphUpdateFunc:[7,1,1,""],MLP:[7,1,1,""],MultiFieldReadout:[7,1,1,""],PairDistance:[7,1,1,""],PairRadialBasisExpansion:[7,1,1,""],PairVector:[7,1,1,""],Pipe:[7,1,1,""],RadialBasisFunctions:[7,1,1,""],ReadOut:[7,1,1,""],ReduceReadOut:[7,1,1,""],Set2Set:[7,1,1,""],SphericalBesselWithHarmonics:[7,1,1,""],StateNetwork:[7,1,1,""],ThreeDInteraction:[7,1,1,""],WeightedReadout:[7,1,1,""],cosine:[7,3,1,""],polynomial:[7,3,1,""]},"m3gnet.layers.AtomNetwork":{call:[7,2,1,""],update_atoms:[7,2,1,""]},"m3gnet.layers.AtomReduceState":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.AtomRef":{call:[7,2,1,""],fit:[7,2,1,""],get_config:[7,2,1,""],inverse_transform:[7,2,1,""],predict_properties:[7,2,1,""],set_property_per_element:[7,2,1,""],transform:[7,2,1,""]},"m3gnet.layers.BaseAtomRef":{call:[7,2,1,""]},"m3gnet.layers.BondNetwork":{call:[7,2,1,""],update_bonds:[7,2,1,""]},"m3gnet.layers.ConcatAtoms":{get_config:[7,2,1,""],update_bonds:[7,2,1,""]},"m3gnet.layers.ConcatBondAtomState":{get_config:[7,2,1,""],update_states:[7,2,1,""]},"m3gnet.layers.Embedding":{call:[7,2,1,""]},"m3gnet.layers.GatedAtomUpdate":{get_config:[7,2,1,""],update_atoms:[7,2,1,""]},"m3gnet.layers.GatedMLP":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.GraphFeaturizer":{get_config:[7,2,1,""]},"m3gnet.layers.GraphFieldEmbedding":{get_config:[7,2,1,""]},"m3gnet.layers.GraphNetworkLayer":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.GraphUpdate":{call:[7,2,1,""]},"m3gnet.layers.GraphUpdateFunc":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.MLP":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.MultiFieldReadout":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.PairDistance":{call:[7,2,1,""]},"m3gnet.layers.PairRadialBasisExpansion":{call:[7,2,1,""],get_config:[7,2,1,""],update_bonds:[7,2,1,""]},"m3gnet.layers.PairVector":{call:[7,2,1,""]},"m3gnet.layers.Pipe":{call:[7,2,1,""],from_config:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.RadialBasisFunctions":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.ReadOut":{call:[7,2,1,""]},"m3gnet.layers.ReduceReadOut":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.Set2Set":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.SphericalBesselWithHarmonics":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.StateNetwork":{call:[7,2,1,""],update_states:[7,2,1,""]},"m3gnet.layers.ThreeDInteraction":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.layers.WeightedReadout":{call:[7,2,1,""],get_config:[7,2,1,""]},"m3gnet.models":{BasePotential:[8,1,1,""],GraphModelMixin:[8,1,1,""],M3GNet:[8,1,1,""],M3GNetCalculator:[8,1,1,""],MolecularDynamics:[8,1,1,""],Potential:[8,1,1,""],Relaxer:[8,1,1,""]},"m3gnet.models.BasePotential":{call:[8,2,1,""],get_ef:[8,2,1,""],get_ef_tensor:[8,2,1,""],get_efs:[8,2,1,""],get_efs_tensor:[8,2,1,""],get_energies:[8,2,1,""],get_forces:[8,2,1,""],get_stresses:[8,2,1,""]},"m3gnet.models.GraphModelMixin":{predict_graph:[8,2,1,""],predict_graphs:[8,2,1,""],predict_structure:[8,2,1,""],predict_structures:[8,2,1,""]},"m3gnet.models.M3GNet":{call:[8,2,1,""],from_config:[8,2,1,""],from_dir:[8,2,1,""],get_config:[8,2,1,""],load:[8,2,1,""],save:[8,2,1,""],set_element_refs:[8,2,1,""]},"m3gnet.models.M3GNetCalculator":{calculate:[8,2,1,""],implemented_properties:[8,4,1,""]},"m3gnet.models.MolecularDynamics":{run:[8,2,1,""],set_atoms:[8,2,1,""]},"m3gnet.models.Potential":{get_energies:[8,2,1,""]},"m3gnet.models.Relaxer":{relax:[8,2,1,""]},"m3gnet.trainers":{PotentialTrainer:[9,1,1,""],Trainer:[9,1,1,""]},"m3gnet.trainers.PotentialTrainer":{train:[9,2,1,""]},"m3gnet.trainers.Trainer":{restart_from_directory:[9,2,1,""],train:[9,2,1,""]},"m3gnet.utils":{Gaussian:[11,1,1,""],SphericalBesselFunction:[11,1,1,""],SphericalHarmonicsFunction:[11,1,1,""],broadcast_states_to_atoms:[11,3,1,""],broadcast_states_to_bonds:[11,3,1,""],check_array_equal:[11,3,1,""],check_shape_consistency:[11,3,1,""],combine_sbf_shf:[11,3,1,""],get_length:[11,3,1,""],get_range_indices_from_n:[11,3,1,""],get_segment_indices_from_n:[11,3,1,""],get_spherical_bessel_roots:[11,3,1,""],register:[11,3,1,""],register_plain:[11,3,1,""],repeat_with_n:[11,3,1,""],reshape_array:[11,3,1,""],spherical_bessel_roots:[11,3,1,""],spherical_bessel_smooth:[11,3,1,""],unsorted_segment_fraction:[11,3,1,""],unsorted_segment_softmax:[11,3,1,""]},"m3gnet.utils.SphericalBesselFunction":{rbf_j0:[11,2,1,""]},m3gnet:{callbacks:[3,0,0,"-"],cli:[4,0,0,"-"],config:[5,0,0,"-"],graph:[6,0,0,"-"],layers:[7,0,0,"-"],models:[8,0,0,"-"],trainers:[9,0,0,"-"],type:[10,0,0,"-"],utils:[11,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"],"5":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute","5":"py:property"},terms:{"0":[1,6,7,8,9,11],"00":1,"0000":1,"001":11,"00645":1,"01":[1,8],"01752":1,"01922":1,"02":1,"02253":1,"02374":11,"02450":1,"0257":1,"02709":1,"0276":1,"02793":1,"02849":1,"03":1,"0303":1,"03084":1,"03265":1,"03454":1,"03893":1,"0399":1,"04":1,"04108":1,"04122":1,"04449":1,"04483":1,"046":1,"0466":1,"05":1,"05053":1,"0583":1,"06":1,"0645":1,"065":1,"0695":1,"06991":1,"07":[1,8],"0772":1,"079":1,"08":[1,6],"08343":1,"08531":1,"09":1,"0959":1,"1":[1,6,8,9,11],"10":[1,6,7,9,11],"100":1,"1000":[1,9],"1072":1,"1093":1,"1096":1,"11":[1,6],"11004":1,"1149":1,"11801":1,"12":[1,6],"123":1,"12493":1,"1265":1,"128":[6,8,9,11],"1281":1,"13":[1,6],"1355":1,"136":1,"1376":1,"14":[1,6],"142":1,"1447":1,"14745":1,"149":1,"15":[1,6],"15275":1,"1548562":1,"155":1,"16":[1,5,6],"16055":1,"16068":1,"16195":1,"1638":1,"16476":1,"166":1,"16702":1,"16762":1,"168":1,"169":1,"16937":1,"17":1,"17129":1,"1713":1,"17262":1,"17431":1,"1789":1,"18":1,"18148":1,"18309":1,"18534":1,"1864":1,"18741":1,"18826":1,"1891":1,"19":1,"1907":11,"1925":1,"19470599":1,"19684":1,"1d":7,"1e":[1,6],"1f":1,"1s":1,"2":[1,6,7,8,11],"20":[1,7,11],"200":9,"2000":1,"2012":1,"2019":1,"2021":[1,8],"2022":[0,1],"20308":1,"20906":1,"20987":1,"21":1,"212":1,"21283":1,"21906":1,"21939":1,"22":1,"2200":1,"2202":1,"22817":1,"23":1,"23376":1,"2358":1,"24":1,"2441":1,"249":1,"25254":1,"25353":1,"25374":1,"25648":1,"2567":1,"25924":1,"26":1,"26212":1,"27":1,"27123":1,"2773":1,"28":1,"283":1,"2835":1,"29":1,"2d":11,"3":[1,6,8,11],"30":[1,11],"300":8,"3000":1,"3008":1,"30383":1,"3058":1,"3087":1,"3090":1,"31":1,"31741":1,"32":[1,5,9],"32052":1,"32221":1,"32229":1,"324209121801212e":8,"328":1,"33":1,"3307":1,"33424":1,"33493":1,"33565":1,"33888":1,"34":1,"34783":1,"35":1,"36":1,"37":1,"38359":1,"392":1,"39634":1,"3d":7,"3f":1,"3x3":1,"4":[1,6,11],"40":11,"4000":1,"41891":1,"41971":1,"42":1,"42398":1,"42682":1,"43":1,"43102":1,"44":1,"44925":1,"45002":1,"45027":1,"4505":1,"451":1,"45247":1,"45297":1,"45474":1,"456":1,"45807":1,"46":1,"462":1,"46873":1,"47":1,"479":1,"48":1,"482":1,"48208":1,"48225":1,"48785":1,"49":1,"49646":1,"499":1,"4993":1,"5":[1,6,11],"50":7,"500":8,"5000":1,"50346":1,"5054":1,"5058":1,"5063":1,"50711":1,"50737":1,"5086":1,"51":1,"51076":1,"51343":1,"51797":1,"51831":1,"52":1,"52597":1,"52739":1,"53":1,"532":1,"53774":1,"538":1,"53838":1,"54":1,"54682":1,"54711":1,"559":1,"56":1,"56111":1,"56567":1,"567":1,"57":1,"5718":1,"57371":1,"577":1,"58":1,"59":1,"597":1,"6":[1,6,8],"60":1,"6000":1,"6084":1,"61":1,"61199":1,"615":1,"61602":1,"61756":1,"61762":1,"62126":1,"62485":1,"626":1,"62619":1,"62745":1,"62905":1,"63":1,"63235":1,"63322":1,"63863":1,"63903":1,"64":1,"64077":1,"64416":1,"65":1,"65023":1,"65131":1,"653":1,"658":1,"66":1,"6605":1,"66148":1,"662":1,"66226":1,"6646":1,"66521":1,"67":1,"67994":1,"68":1,"69":1,"69169":1,"69497":1,"69553":1,"69625":1,"698":1,"69873":1,"7":[1,6],"7000":1,"70308":1,"70797":1,"71921":1,"72":1,"72044":1,"721":1,"72434":1,"72451":1,"73":1,"73027":1,"731":1,"73376":1,"73892":1,"744":1,"75018":1,"75055":1,"76":1,"76286":1,"767":1,"76843":1,"7698":1,"77":1,"78":1,"79":1,"8":[1,6,8],"80":1,"8000":1,"81":1,"82":1,"83041":1,"832":1,"83451":1,"838":1,"84":1,"84005":1,"8439":1,"8456":1,"84935":1,"84993":1,"85":1,"850":1,"85237":1,"856":1,"859":1,"869":1,"87":1,"87403":1,"87573":1,"8770":1,"87716":1,"88":1,"8804":1,"89":1,"8908":1,"9":[1,6],"9000":1,"90089":1,"9016":1,"90604":1,"90767":1,"912":1,"9176":1,"92":1,"92943":1,"9362":1,"9380":1,"94":1,"94083":1,"9419":1,"94481":1,"94513":1,"95466":1,"95673":1,"95707":1,"95744":1,"96":1,"96119":1,"9679":1,"97":1,"9702":1,"97677":1,"97714":1,"98":1,"98219":1,"98805":1,"9885":1,"99":1,"99254":1,"99346":1,"\u00e5":1,"abstract":[6,8],"boolean":6,"case":[0,1],"class":[0,3,5,6,7,8,9,11],"default":[0,1,6,8],"do":[1,11],"final":1,"float":[1,6,7,8,9,11],"function":[3,5,7,8,9,11],"import":1,"int":[3,6,8,9,11],"new":[1,6,7,8],"public":1,"return":[3,5,6,7,8,9,11],"static":[6,11],"true":[1,6,8,9,11],"try":1,A:[1,7],As:1,At:1,For:[1,7,11],If:[1,8,9,11],In:[1,6],It:[1,7],Of:1,On:[7,11],One:1,The:[1,2,6,7,8,11],With:1,_atom:7,_atom_ref:7,_base:[7,8,9],_batch:6,_bond:7,_convert:6,_core:7,_gn:7,_m3gnet:[8,9],_readout:7,_state:7,_type:[6,8],ab:1,abc:[1,8],ac02:1,ac:1,accomplish:1,accord:[6,11],accuraci:1,achiev:1,aci:1,across:1,activ:[1,7],ad:6,adam:1,add:[0,6,7,8],addit:1,after:[1,11],ag:1,ai:1,al:1,ala:1,all:[1,6],almost:1,alp:1,alreadi:1,alsb:1,also:[1,7],an:[1,6,7,9,11],anaconda:1,angl:[6,7],angular:7,ani:[1,11],append_trajectori:8,appli:8,applic:1,ar:[1,6,7],arbitrari:1,architectur:1,arg:[3,4,5,6,7,8,9,11],arrai:[6,7,11],array1:11,array2:11,art:1,arxiv:[1,11],as_list:6,as_tf:6,ase:[1,6,8],assemble_material_graph:6,assum:6,atom:[1,6,7,8,11],atom_posit:6,atomnetwork:7,atomreducest:7,atomref:7,attribut:[6,7,11],attributeupdatemixin:6,au:1,augment:11,auto:1,avail:[1,6],averag:7,axi:11,azimuth:6,ba:1,bar:7,base:[1,3,5,6,7,8,9,11],base_attribut:6,base_lay:[6,7,8],baseatomref:7,basegraphconvert:6,basepotenti:8,basi:[7,11],basic:[1,6],batch:[1,3,6,8,9],batch_siz:[6,8,9],bcc:1,becom:11,been:1,below:1,benchmark:1,bessel:[7,11],better:1,between:5,binari:1,blend:1,block:11,bn:1,bodi:[1,7],bond:[0,6,7,11],bond_atom_indic:6,bond_weight:6,bondnetwork:7,bool:[6,8,9,11],both:7,boundari:[6,11],bp:1,broadcast:11,broadcast_states_to_atom:11,broadcast_states_to_bond:11,bug:0,c:1,ca:1,caesium:1,calcul:[1,6,7,8,11],call:[1,3,7,8],callback:[2,9,13],can:[1,8],candid:1,capabl:1,categor:7,categori:1,cavo3:1,cd:1,cdse:1,cdte:1,ce:1,center:[7,11],center_indic:6,ch11231:1,chang:[7,8],check:[9,11],check_array_equ:11,check_shape_consist:11,chemic:1,chen:1,chi:1,chlorid:1,choos:5,cif:1,cite:1,cl:11,class_nam:11,classmethod:[5,6,7,8],classvar:6,cli:[0,2,13],clip:9,clip_norm:9,close:1,cluster:1,code:[0,3],collat:6,column:11,combin:[7,9,11],combinatori:1,combine_sbf_shf:11,command:4,common:1,comparison:1,compli:11,comprehens:1,compress:1,compressibility_au:8,comput:[1,6,7,8,11],compute_stress:8,concatatom:7,concatbondatomst:7,conda:1,config:[2,7,8,13],configur:7,consecut:7,constant:1,construct:[6,7,8],contain:[1,11],content:13,continu:[7,9],contract:1,conveni:10,convent:1,converg:8,convert:[6,7],convert_mani:6,coordin:1,copi:[6,7],core:[1,6,8],correct:7,cosin:7,could:7,cours:1,cr:1,creat:1,crn:1,crystal:[0,1],cs:1,cscl:1,csf:1,csi:1,cu:1,cubic:1,custom:[8,11],custom_object:8,cutoff:[6,7,11],cython:1,data:[1,5,6,9,11],data_typ:5,data_util:6,databas:1,datatyp:5,de:1,deep:1,defin:[8,10],dep:1,depart:1,depend:1,deprecationwarn:1,deriv:11,detail:1,develop:1,dft:1,dgaines2:0,diamond:1,dict:[3,6,7,8],dictionari:[6,7,8],differ:[1,11],differenti:1,dimens:[7,11],dimension:[7,11],directli:1,directori:[3,8,9],dirnam:[8,9],discoveri:1,distanc:[6,7,11],distribut:7,divers:1,divis:1,document:[0,1],doe:[1,7,11],doi:1,done:1,download:[0,1],due:1,dure:9,dynam:8,e:[7,9,11],each:[1,6,7,11],earli:[0,9],early_stop_pati:9,eas:1,edg:[0,6,7],ef:8,eij:7,either:8,ekin:1,element:[1,7,8,9],element_ref:8,embed:7,enabl:0,encount:1,end:[3,6,11],energi:[1,6,7,8,9],engin:[1,6,7,8],ensembl:[1,8],ensur:[1,11],entir:1,entri:11,environ:1,epoch:[1,3,6,9],epot:1,equal:11,equival:11,error:[1,11],etc:[1,6,7,9],etot:1,eu:1,eutio3:1,ev:1,exampl:[1,7,11],exp:1,expand:[7,11],expans:[1,7,11],expect:1,experiment:1,expt:1,extens:[7,8],extrem:1,f:1,fact:11,factor:1,fals:[1,8,9,11],faster:11,fcc:1,fe:1,featur:[1,6,7],field:[6,7,9,11],figshar:1,file:[1,3,8],filter:0,filterwarn:1,final_energy_per_atom:1,final_structur:1,fire:8,first:[0,1,11],fit:7,fit_per_element_offset:[1,9],fix:[0,1,6],flexibl:[0,1],float16:5,float32:5,fmax:8,folder:1,follow:[1,6,7,8],forc:[1,6,8,9],force_loss_ratio:9,form:6,format:[1,6,7],foundat:1,framework:[1,2,6,7,8,11],free_energi:8,from:[1,4,6,7,8,9,11],from_config:[7,8],from_dir:8,from_list:6,full:6,fund:1,futur:1,g:9,ga:1,gaa:1,gap:1,gasb:1,gase:1,gate:7,gatedatomupd:7,gatedmlp:7,gaussian:11,gb:1,ge:1,gener:[1,6],get:[6,7,8,11],get_atom_featur:6,get_config:[6,7,8],get_ef:8,get_ef_tensor:8,get_efs_tensor:8,get_energi:8,get_fixed_radius_bond:6,get_forc:8,get_length:11,get_pair_vector_from_graph:6,get_range_indices_from_n:11,get_segment_indices_from_n:11,get_spherical_bessel_root:11,get_stat:6,get_stress:8,github:1,give:[1,11],given:[6,8],gpa:1,gpu:1,gradient:9,grant:1,graph:[1,2,7,8,9,11,13],graph_list:[6,8],graph_with_position_attribut:6,graphfeatur:7,graphfieldembed:7,graphmodelmixin:8,graphnetworklay:7,graphs_or_structur:9,graphupd:7,graphupdatefunc:7,h2o:7,h:7,ha:[1,6,11],halit:1,hand:11,handl:[4,8],harmon:[7,11],has_threebodi:6,have:1,help:1,here:[1,8,11],hfc0:1,hfn:1,host:1,howev:1,http:[1,11],i:[7,11],iap:1,icsd:1,id:[3,11],ignor:1,imag:6,immedi:1,implement:[1,7,11],implemented_properti:8,ina:1,includ:[1,7,8],include_forc:8,include_stress:8,include_threebody_indic:6,incorpor:1,index:[6,11],indic:[6,7,11],infer:1,infil:1,info:7,inform:[1,7],init:1,initi:[0,1,7],inp:1,input:7,insb:1,instanc:[6,8,11],int32:5,integ:[6,11],interact:[1,7],interatom:1,interest:1,interv:[1,8],inverse_transform:7,io:8,iodid:1,ionic:1,ir:1,is_intens:1,isoval:1,item:11,its:1,itself:1,j0:11,j:11,j_l:11,janosh:0,jupyt:1,just:1,k:1,kbar:1,kbr:1,kc23mp:1,kcl:1,kei:1,kera:[1,3,6,7,8,9],kf:1,ki:1,know:11,kr:1,ktao3:1,kwarg:[6,7,8,11],l:11,laptop:1,last:7,later:7,latter:7,lattic:[1,6,7],layer:[2,6,8,13],learn:1,learnabl:7,len:1,length:6,less:[0,1],li2o:1,li2o_relax:1,li:1,libr:1,licl:1,lif:1,lifepo4:1,lii:1,like:1,limit:1,linalg:11,list:[1,6,7,8,9,11],load:[1,8],log:3,logfil:[1,8],logic:7,loginterv:[1,8],look:1,loss:9,lower:[1,7],m1:1,m3g:1,m3gnetcalcul:8,m9:1,m:11,mac:1,machin:1,maco:1,mai:[1,6,11],main:[4,7,8],major:1,mani:[1,6],manual:[1,3],manualstop:3,match:11,materi:[1,6,8,9],materialgraph:[6,7,8,9],materialgraphbatch:6,materialgraphbatchenergyforcestress:6,matric:[1,6],matrix:[1,11],max:[1,8,11],max_l:11,max_n:11,md:[0,1,8],mean:7,mean_squared_error:9,megnet:1,memori:1,metal:1,method:[5,8],metric:9,mgo:1,million:1,miniconda:1,minim:1,minut:1,ml:1,mlp:7,mlp_:7,mo:1,model:[0,2,9,13],model_nam:8,modul:[1,13],molecul:[6,8],molecular:[0,8],moleculardynam:[0,1,8],monitor:[8,9],more:[0,1],mp:[1,8],mpf:1,multi:7,multifieldreadout:7,multipli:1,n:[1,11],n_atom:6,n_bond:6,n_struct:6,n_triple_:6,n_triple_i:6,n_triple_ij:6,na:[1,8,11],nabr:1,nacl:1,naf:1,nai:1,name:[1,6,8,9,11],narrai:11,nation:1,nb:[1,11],nbn:1,ndarrai:[6,7,8,11],ne:1,necessari:1,need:[1,6],neg:1,neighbor:7,neighbor_indic:6,network:[1,7,9],neural:[1,9],ni:1,nobl:1,node:6,noisi:0,non:1,none:[3,5,6,8,9,11],norm:[9,11],note:1,notebook:1,now:1,np:[6,7,8,11],np_float:5,np_int:5,npt:0,ns:[8,11],nstate:11,num_seg:11,number:[1,6,7,8,11],numer:6,numerical_tol:6,numpi:[5,6,8,11],nvt:[1,8],nx3:1,o:7,obj:8,object:[5,6,7,8,9,11],observ:1,obtain:[1,6],offic:1,offici:0,offset:[6,7,9],on_batch_end:3,on_epoch_end:6,onc:1,one:[1,6],ong:1,onli:1,onto:7,op:[6,7,8,11],oper:6,optim:[1,8,9],optimizer_v2:9,optimizerv2:9,option:[1,3,6,8,9,11],order:11,org:[1,11],origin:[7,11],orthogon:11,other:[1,11],otherwis:1,our:1,out:7,outfil:1,output:[0,1,7,8],overwritten:6,own:1,oxid:1,packag:13,pad:6,page:1,pair:[6,7,11],pairdist:7,pairradialbasisexpans:7,pairvector:7,paramet:[1,4],pass:1,path:8,patienc:9,pauciti:1,pb:1,pbc_offset:6,pbte:1,pd:1,pdf:11,per:7,perceptron:7,perform:[1,7],period:[1,6,7],perovskit:1,phi:[6,11],ping:1,pip:1,pipe:7,pleas:1,point:[1,9],polar:6,polynomi:7,posit:[6,7,8],potenti:[0,1,8,9],potentialtrain:[1,9],power:1,pre:[1,8,11],precis:11,precomput:11,predict:[1,7,8,9],predict_graph:8,predict_properti:7,predict_structur:8,preprint:1,pressur:8,pretrain:8,previou:[1,8,9],primarili:1,prime:7,print:1,prior:1,pro:1,program:1,progress:9,project:1,proper:7,properti:[1,6,7,8,9],property_per_el:7,protobuf:1,prototyp:1,provid:1,ps:1,pt:1,put:3,pymatgen:[0,1,6,8],pytest:1,python:[1,6,7,8,11],quantiti:1,quaternari:1,quick:1,quit:1,r:[7,11],radial:7,radialbasisfunct:7,radiu:[6,7,11],radiuscutoffgraphconvert:6,ran:1,rang:11,ratio:9,rb:1,rbbr:1,rbcl:1,rbf:[1,7],rbf_j0:11,rbi:1,read:7,readout:7,recommend:1,record:1,reduc:7,reducereadout:7,ref:11,refer:[7,8],regist:11,register_plain:11,rel:11,relax:[0,8],relax_cel:8,relax_result:1,relax_structur:4,releas:0,remov:0,repeat:11,repeat_with_n:11,report:11,repr:[7,8],represent:[1,6,7],reproduc:1,requir:0,reshap:11,reshape_arrai:11,restart_from_directori:9,result:[8,11],rh:1,right:1,root:11,row:7,rtol:11,rtx:1,run:[1,7,8],s:1,same:1,save:[1,8,9],save_checkpoint:[1,9],sbf:11,scale:1,scheme:0,scienc:1,scn:1,second:[1,11],section:1,see:1,segment:11,segment_id:11,select:1,sequenc:[6,7,11],sequenti:7,serial:[6,7,8],set2set:7,set:[5,6,7,8],set_atom:8,set_default_st:6,set_dtyp:5,set_element_ref:8,set_global_dtyp:5,set_property_per_el:7,sever:[7,10],shape:[6,11],shf:11,should:[1,8],show:9,shown:1,shuffl:6,shyue:1,si:1,sigmoid:7,similar:[1,7],similarli:1,simpl:[0,1,7,11],simul:1,sinc:[1,11],singl:[1,6],size:[1,8,9,11],smooth:11,softmax:11,sourc:1,space:[1,7],special:1,specifi:11,speed:1,spheric:[7,11],spherical_bessel_root:11,spherical_bessel_smooth:11,sphericalbesselfunct:11,sphericalbesselwithharmon:7,sphericalharmonicsfunct:11,sr:1,srtio3:1,srvo3:1,stabl:1,standard:1,state:[1,6,7,11],statenetwork:7,stdout:1,step:[1,7,8],stop:[0,1,3,9],str:[5,6,8,9],strang:1,stress:[0,1,6,8,9],stress_loss_ratio:9,stress_weight:8,stretch:1,structs_or_graph:7,structur:[6,7,8,9],structureormolecul:[6,8],subclass:8,submodul:13,subpackag:13,substitut:1,subtract:7,successfulli:1,sum:[7,8,9],supercomput:1,support:[0,1],surrog:1,sympi:11,synthes:1,system_chang:8,t:[1,11],ta:1,tac0:1,take:[1,7,11],target:[6,9],taup:8,taut:8,temperatur:[1,8],tensor:[6,7,8,11],tensorflow:[1,5,6,7,8,11],tensori:1,ternari:1,test:1,tf:[1,6,7,8,9,11],tf_compute_distance_angl:6,tf_float:5,tf_int:5,th:[1,7,11],than:1,them:7,thermodynam:1,theta:6,thi:[1,6,7,8,11],thin:[7,8],thing:7,three:7,three_basi:7,three_body_attribut:6,three_cutoff:7,threebodi:6,threebody_cutoff:6,threedinteract:7,through:7,tic:1,time:[1,11],timestep:[1,8],tin:1,toler:[6,8,11],tool:0,top:7,total:8,train:[0,3,6,8,9],train_metr:9,trainer:[1,2,13],traj:1,traj_fil:8,trajectori:[1,8],transform:7,tripl:[0,6],triple_bond_indic:6,triple_bond_length:6,triplet:6,tupl:[6,8],two:11,type:[2,5,6,13],u:[1,7],ultra:1,under:1,uninstal:1,union:[6,8,11],unit:1,univers:1,unsort:11,unsorted_segment_fract:11,unsorted_segment_softmax:11,up:[1,7],updat:7,update_atom:7,update_bond:7,update_st:7,us:[1,5,6,7,10,11],use_phi:11,user:1,userwarn:1,util:[2,6,13],v:[1,7],val:9,val_acc:9,val_auc:9,val_energi:[1,9],val_forc:[1,9],val_ma:9,val_metr:9,val_monitor:9,val_stress:[1,9],val_structur:1,valid:[1,9],validation_graphs_or_structur:[1,9],validation_target:9,valu:[1,6,7,8,11],vanish:11,vasp:1,vc0:1,vector:[6,7],verbos:[0,1,8,9],version:1,vi:7,via:[1,7],view:4,vj:7,vn:1,vs:1,vvvlzy:0,w:1,wa:[1,11],warn:[0,1],we:[1,11],weight:[0,7,8,11],weightedreadout:7,well:1,were:8,where:[1,7],whether:[6,8,9,11],which:[1,8],width:11,wikipedia:1,wise:9,wish:1,within:[0,1,6],without:6,work:1,wrapper:[7,8],write:1,written:1,wrong:1,x:[7,11],xcode:1,xe:1,xsede:1,yb:1,yet:1,you:1,your:1,z_:11,zenodo:0,zero:11,zinc:1,zn:1,zno:1,zrc0:1,zrn:1},titles:["Changelog","M3GNet","m3gnet package","m3gnet.callbacks module","m3gnet.cli module","m3gnet.config module","m3gnet.graph package","m3gnet.layers package","m3gnet.models package","m3gnet.trainers package","m3gnet.type module","m3gnet.utils package","m3gnet.utils.constants module","m3gnet"],titleterms:{"0":0,"1":0,"2":0,"4":0,"6":0,"7":0,"8":0,"9":0,acknowledg:1,api:1,appl:1,callback:3,chang:1,changelog:0,cli:[1,4],config:5,constant:12,content:[1,2,6,7,8,9,11],dataset:1,doc:1,dynam:1,graph:6,instal:1,layer:7,log:1,m3gnet:[1,2,3,4,5,6,7,8,9,10,11,12,13],mattervers:1,model:[1,8],modul:[2,3,4,5,6,7,8,9,10,11,12],molecular:1,packag:[2,6,7,8,9,11],refer:1,relax:1,requir:1,silicon:1,structur:1,submodul:2,subpackag:2,system:1,tabl:1,tool:1,train:1,trainer:9,type:10,usag:1,util:[11,12],v0:0}})