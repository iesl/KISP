import tensorflow as tf
from peach.tf_nn.attn import compatibility_fn, direct_mask_generation, split_head, combine_head
import peach.tf_nn.nn
from peach.tf_nn.nn import residual_connection, residual_connection_with_dense, bn_dense_layer_v2, alt_lin_layer1, dropout, \
    bn_dense_layer_multi_head
from peach.tf_nn.lnorm import layer_normalize    
from peach.tf_nn.general import exp_mask_v3, mask_v3, act_name2fn, get_shape_list
import logging
import sys
import imp



# for module in sys.modules.values():
#     imp.reload(module)





def transformer_seq_decoder(
        dec_input_emb_mat, decoder_ids, encoder_states, decoder_mask, encoder_mask, n_out_channel, num_layers,
        decoder_history_inputs=None,
        hn=768, head_num=12, act_name="gelu",
        wd=0., is_training=None, keep_prob_dense=1., keep_prob_attn=1., keep_prob_res=1.,
        scope=None, mode=None, method=None, entity_vec_list=None, entity_pos_list=None, num_vec_list=None,
        num_pos_list = None, pred_vec_list=None, type_vec_list=None, one_hop_list=None, one_hop_mask = None, input_pos_ids=None, 
        mh_queries=None, mh_counts=None, mh_mask=None, bmh_queries=None, bmh_counts=None, bmh_mask=None, dec_query_result_matrix=None, dec_bquery_result_matrix=None, cfg=None):

    

    with tf.variable_scope(scope or "transformer_seq_decoder"):
        # dec_input_emb_mat = tf.Print(dec_input_emb_mat, ['dec_input_emb_mat', dec_input_emb_mat, tf.shape(dec_input_emb_mat)], first_n=10)
        # decoder_ids = tf.Print(decoder_ids, ['decoder_ids', decoder_ids, tf.shape(decoder_ids)], first_n=10)
        # encoder_states = tf.Print(encoder_states, ['encoder_states', encoder_states, tf.shape(encoder_states)], first_n=10)
        # encoder_mask = tf.Print(encoder_mask, ['encoder_mask', encoder_mask, tf.shape(encoder_mask)], first_n=10)
        # n_out_channel = tf.Print(n_out_channel, ['n_out_channel', n_out_channel, tf.shape(n_out_channel)], first_n=10)
        # num_layers = tf.Print(num_layers, ['num_layers', num_layers, tf.shape(num_layers)], first_n=10)
        # hn = tf.Print(hn, ['hn', hn, tf.shape(hn)], first_n=10)
        # head_num = tf.Print(head_num, ['head_num', head_num, tf.shape(head_num)], first_n=10)
        # act_name = tf.Print(act_name, ['act_name', act_name, tf.shape(act_name)], first_n=10)
        # wd = tf.Print(wd, ['wd', wd, tf.shape(wd)], first_n=10)
        # is_training = tf.Print(is_training, ['is_training', is_training, tf.shape(is_training)], first_n=10)

        # bs,sl,hn
        # bs, sl, int => bs,sl,hn

        # if mode == 'alt_train':
        #     decoder_ids = tf.Print(decoder_ids, \
        #         ['going into training', 'ent_vec', entity_vec_list, tf.shape(entity_vec_list),'ent_pos', entity_pos_list, tf.shape(entity_pos_list)], first_n=10)
        #     # TODO bring mask into here...combine entity and dec_matrix
        # elif mode == 'parallel_test':
        #     decoder_ids = tf.Print(decoder_ids, \
        #         ['going into testing', 'ent_vec', entity_vec_list, tf.shape(entity_vec_list),'ent_pos',  tf.shape(entity_pos_list)], first_n=10)



        mh_overall_queries = mh_queries
        bmh_overall_queries = bmh_queries

        with tf.variable_scope("decoder_emb"):
            decoder_inputs = tf.nn.embedding_lookup(dec_input_emb_mat, decoder_ids) # bs,sl,hn #skeleton

        if mode == 'alt_train':
            if(method == 'Nprojectup_5_all' or method == "Nprojectup_5_attend_oh" or "path" in method):
                
                # max_seq_len_num = tf.shape(encoder_states)[-2]
                replaced_num_pos = tf.where(tf.greater_equal(num_pos_list, 0), num_pos_list, tf.zeros_like(num_pos_list))
                num_emb = tf.batch_gather(encoder_states, replaced_num_pos)*tf.cast(tf.expand_dims(tf.not_equal(num_pos_list, -1),-1), tf.float32)

                # max_seq_len = tf.shape(encoder_states)[-2]
                replaced_pos = tf.where(tf.greater_equal(entity_pos_list, 0), entity_pos_list, tf.zeros_like(entity_pos_list))
                eent_emb = tf.batch_gather(encoder_states, replaced_pos)*tf.cast(tf.expand_dims(tf.not_equal(entity_pos_list, -1),-1), tf.float32)


                # max_seq_len = tf.shape(entity_vec_list)[-2]
                replaced_pos1 = tf.where(tf.greater_equal(entity_pos_list, 0), entity_pos_list, tf.zeros_like(entity_pos_list))
                replaced_pos2 = tf.batch_gather(input_pos_ids, replaced_pos1)
                ent_emb = tf.batch_gather(entity_vec_list, replaced_pos2)*tf.cast(tf.expand_dims(tf.not_equal(entity_pos_list, -1),-1), tf.float32)


                kg_embed = tf.concat([ent_emb, eent_emb, num_emb, pred_vec_list, type_vec_list],axis=-1)
                # kg_embed = tf.Print(kg_embed, [kg_embed.shape], first_n=10)
                kg_emb_new = bn_dense_layer_v2(  # bs,slt,hd_dim * hd_num
                kg_embed, hn, False, 0., 'projectup_maxpool',
                'linear', False, wd, keep_prob_dense, is_training, dup_num=1, merge_var=True)   

                pooled_inputs = tf.reduce_max([decoder_inputs, kg_emb_new],axis=0)
                decoder_inputs = pooled_inputs
                print("Method called is "+method+"with SKI##", flush=True)

            elif(method == 'Mprojectup_5_all' or method=='Mprojectup_5_attend_oh'):
                

                # max_seq_len_num = tf.shape(encoder_states)[-2]

                replaced_num_pos = tf.argmax(tf.cast(tf.equal(tf.expand_dims(input_pos_ids, axis=-2), tf.expand_dims(num_pos_list, axis=-1)), tf.int32), axis=-1)
                num_emb = tf.batch_gather(encoder_states, tf.cast(replaced_num_pos, tf.int32))*tf.cast(tf.expand_dims(tf.not_equal(num_pos_list, -1),-1), tf.float32)

                # max_seq_len = tf.shape(encoder_states)[-2]
                replaced_ent_pos = tf.argmax(tf.cast(tf.equal(tf.expand_dims(input_pos_ids, axis=-2), tf.expand_dims(entity_pos_list, axis=-1)), tf.int32), axis=-1)
                eent_emb = tf.batch_gather(encoder_states, tf.cast(replaced_ent_pos, tf.int32))*tf.cast(tf.expand_dims(tf.not_equal(entity_pos_list, -1),-1), tf.float32)


                # max_seq_len = tf.shape(entity_vec_list)[-2]
                # replaced_pos1 = tf.where(tf.greater_equal(entity_pos_list, 0), entity_pos_list, tf.zeros_like(entity_pos_list))
                # replaced_pos2 = tf.batch_gather(input_pos_ids, replaced_pos1)

                # max_seq_len = tf.shape(entity_vec_list)[-2]
                # replaced_pos = tf.where(tf.greater_equal(entity_pos_list, 0), entity_pos_list, tf.ones_like(entity_pos_list)*(max_seq_len-1))
                ent_emb = tf.batch_gather(entity_vec_list, entity_pos_list)*tf.cast(tf.expand_dims(tf.not_equal(entity_pos_list, -1),-1), tf.float32)


                ent_emb = tf.Print(ent_emb, \
                    ["entity_pos_list", tf.shape(entity_pos_list), "input_pos_ids", tf.shape(input_pos_ids), \
                        "encoder_states", tf.shape(encoder_states), "entity_vec_list", tf.shape(entity_vec_list), entity_vec_list,\
                            entity_pos_list, ent_emb], summarize=1000, first_n=5)

                ent_emb = tf.Print(ent_emb,["ent_emb", tf.shape(ent_emb)], first_n=5, summarize=10000)

                # max_seq_len = tf.shape(entity_vec_list)[-2]
                # input_pos_ids = tf.concat([input_pos_ids, tf.ones_like(input_pos_ids[:, 0:1])*(max_seq_len-1)], axis=-1)
                # max_pos_len = tf.shape(input_pos_ids)[-1]
                # m1_epos_list = entity_pos_list
                # entity_pos_list = tf.where(tf.greater_equal(entity_pos_list, 0), entity_pos_list, tf.ones_like(entity_pos_list)*(max_pos_len-1))
                # replaced_pos = tf.batch_gather(input_pos_ids, entity_pos_list)
                # ent_emb = tf.batch_gather(entity_vec_list, replaced_pos)*tf.cast(tf.expand_dims(tf.not_equal(m1_epos_list, -1),-1), tf.float32)

                # ent_emb = tf.Print(ent_emb, ["entity_pos_list", entity_pos_list, "num_pos_list", num_pos_list,\
                #      "num_emb", tf.shape(num_emb), "eent_emb",\
                #      tf.shape(eent_emb), eent_emb[0:3,0:5]], first_n=5, summarize=10000)

                ent_emb = tf.Print(ent_emb,["ent_emb", tf.shape(ent_emb)], first_n=5, summarize=10000)

                # max_seq_len = tf.shape(entity_vec_list)[-2]
                # input_pos_ids = tf.concat([input_pos_ids, tf.ones_like(input_pos_ids[:, 0:1])*(max_seq_len-1)], axis=-1)
                # max_pos_len = tf.shape(input_pos_ids)[-1]
                # m1_epos_list = entity_pos_list
                # entity_pos_list = tf.where(tf.greater_equal(entity_pos_list, 0), entity_pos_list, tf.ones_like(entity_pos_list)*(max_pos_len-1))
                # replaced_pos = tf.batch_gather(input_pos_ids, entity_pos_list)
                # ent_emb = tf.batch_gather(entity_vec_list, replaced_pos)*tf.cast(tf.expand_dims(tf.not_equal(m1_epos_list, -1),-1), tf.float32)


                kg_embed = tf.concat([ent_emb, eent_emb, num_emb, pred_vec_list, type_vec_list],axis=-1)
                # kg_embed = tf.Print(kg_embed, [kg_embed.shape], first_n=10)
                kg_emb_new = bn_dense_layer_v2(  # bs,slt,hd_dim * hd_num
                kg_embed, hn, False, 0., 'projectup_maxpool',
                'linear', False, wd, keep_prob_dense, is_training, dup_num=1, merge_var=True)   

                pooled_inputs = tf.reduce_max([decoder_inputs, kg_emb_new],axis=0)
                decoder_inputs = pooled_inputs
                logging.info("Method called is "+method)

            else:
                logging.info("Alternate training called but method not specific - not using entity embeddings")
        else:
            logging.info("Normal training without Entity embeddings called")

    
        # hn after this point refers to the hidden dimension
        with tf.variable_scope("decoder_recurrence"):
            dec_outputs, new_decoder_history_inputs = transformer_decoder(  # bs,sl,hn
                decoder_inputs, encoder_states, decoder_mask, encoder_mask, num_layers,
                decoder_history_inputs,
                hn, head_num, act_name,
                wd, is_training, keep_prob_dense, keep_prob_attn, keep_prob_res,
                scope="transformer_decoder", one_hop_list = one_hop_list, one_hop_mask = one_hop_mask, mh_overall_queries=mh_overall_queries,
                mh_mask=mh_mask, bmh_overall_queries=bmh_overall_queries, bmh_mask=bmh_mask, method=method
            )
            # prediction logits: two layer
            # pre_logits_seq2seq = bn_dense_layer_v2(
            #     dec_outputs, hn, True, 0., "pre_logits_seq2seq", act_name,
            #     False, 0., keep_prob_dense, is_training
            # )
            # print('out channel problem', n_out_channel, flush=True)
            logits_seq2seq = bn_dense_layer_v2(  # bs,sl,
                dec_outputs, n_out_channel, True, 0., "logits_seq2seq", "linear",
                False, 0., keep_prob_dense, is_training
            )
            return dec_outputs, logits_seq2seq, new_decoder_history_inputs


def transformer_decoder(
        decoder_input, encoder_output,
        decoder_mask, encoder_mask,
        num_layers, decoder_history_inputs=None,
        hn=768, head_num=12, act_name="gelu",
        wd=0., is_training=None, keep_prob_dense=1., keep_prob_attn=1., keep_prob_res=1.,
        scope=None, one_hop_list = None , one_hop_mask=None, mh_overall_queries=None,
        mh_mask=None, bmh_overall_queries=None, bmh_mask=None, method=None
):
    fwd_mask = direct_mask_generation(decoder_mask, direct="forward", attn_self=True)  # DONE: double check this

    use_decoder_history = False
    decoder_history_inputs_list = []

    # we supply None to decoder_history_inputs
    if not isinstance(decoder_history_inputs, type(None)):
        use_decoder_history = True
        decoder_history_inputs_list = tf.unstack(decoder_history_inputs, num_layers, axis=1)
        fwd_mask = None
    cur_history_inputs_list = []

    with tf.variable_scope(scope or "transformer_decoder"):
        x = decoder_input
        # num_layers is decoder layer = 2
        for layer in range(num_layers):
            with tf.variable_scope("layer_{}".format(layer)):
                tensor_to_prev = decoder_history_inputs_list[layer] if use_decoder_history else None
                cur_history_inputs_list.append(x)
                with tf.variable_scope("self_attention"):
                    y = multihead_attention_decoder(
                        x, x, decoder_mask, fwd_mask, "linear",
                        hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        tensor_to_prev=tensor_to_prev, mask_prev_to=None,
                    )

                    x = residual_connection(x, y, is_training, keep_prob_res)
                with tf.variable_scope("encdec_attention"):
                    y = multihead_attention_decoder(
                        x, encoder_output, encoder_mask, None, "linear",
                        hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                    )

                    x = residual_connection(x, y, is_training, keep_prob_res)

                if(method=="Nprojectup_5_attend_oh" or method=="Mprojectup_5_attend_oh"):
                    with tf.variable_scope("one_hop_attention"):
                        y = multihead_attention_decoder(
                            x, one_hop_list, one_hop_mask, None, "linear",
                            hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        )

                        # y = tf.contrib.layers.layer_norm(y, begin_norm_axis=2)

                        x = residual_connection(x, y, is_training, keep_prob_res)
                        x=tf.Print(x,["x", tf.shape(x), "one hop in use"], first_n=5)
                        print("Method called is "+method+" first", flush=True)
                elif(method=="path_attend_5all"):
                    with tf.variable_scope("one_hop_attention"):
                        y = multihead_attention_decoder(
                            x, one_hop_list, one_hop_mask, None, "linear",
                            hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        )

                        # y = tf.contrib.layers.layer_norm(y, begin_norm_axis=2)

                        x = residual_connection(x, y, is_training, keep_prob_res)

                    mh_overall_queries = tf.Print(mh_overall_queries, [tf.shape(mh_overall_queries), tf.shape(mh_mask), tf.shape(one_hop_list), tf.shape(one_hop_mask)], first_n=10)
                    with tf.variable_scope("mh_path_attention"):
                        y = multihead_attention_decoder(
                            x, mh_overall_queries, mh_mask, None, "linear",
                            hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        )
                        x = residual_connection(x, y, is_training, keep_prob_res)
                    print("Method called is "+method+" path_attend_5all", flush=True)
                elif(method=="path_attend_5only"):
        
                    mh_overall_queries = tf.Print(mh_overall_queries, [tf.shape(mh_overall_queries), tf.shape(mh_mask), tf.shape(one_hop_list), tf.shape(one_hop_mask)], first_n=10)
                    with tf.variable_scope("mh_path_attention"):
                        y = multihead_attention_decoder(
                            x, mh_overall_queries, mh_mask, None, "linear",
                            hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        )

                        x = residual_connection(x, y, is_training, keep_prob_res)
                    x = tf.Print(x, [tf.shape(x), tf.shape(y), tf.shape(one_hop_list), tf.shape(one_hop_mask)], first_n=10)
                    print("Method called is "+method+" path_attend_5only", flush=True)
                elif(method=="path_attend_5bs_only"):
        
                    mh_overall_queries = tf.Print(mh_overall_queries, [tf.shape(mh_overall_queries), tf.shape(mh_mask), tf.shape(one_hop_list), tf.shape(one_hop_mask)], first_n=10)
                    with tf.variable_scope("mh_path_attention"):
                        y = multihead_attention_decoder(
                            x, mh_overall_queries, mh_mask, None, "linear",
                            hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        )

                        x = residual_connection(x, y, is_training, keep_prob_res)
                    with tf.variable_scope("bmh_path_attention"):
                        y = multihead_attention_decoder(
                            x, bmh_overall_queries, bmh_mask, None, "linear",
                            hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        )

                        x = residual_connection(x, y, is_training, keep_prob_res)

                    x = tf.Print(x, [tf.shape(x), tf.shape(y), tf.shape(one_hop_list), tf.shape(one_hop_mask)], first_n=10)
                    print("Method called is "+method+" path_attend_5bs_only", flush=True)
                elif(method=="path_attend_5bs_all"):
                    with tf.variable_scope("one_hop_attention"):
                        y = multihead_attention_decoder(
                            x, one_hop_list, one_hop_mask, None, "linear",
                            hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        )

                        x = residual_connection(x, y, is_training, keep_prob_res)


                    mh_overall_queries = tf.Print(mh_overall_queries, [tf.shape(mh_overall_queries), tf.shape(mh_mask), tf.shape(one_hop_list), tf.shape(one_hop_mask)], first_n=10)
                    with tf.variable_scope("mh_path_attention"):
                        y = multihead_attention_decoder(
                            x, mh_overall_queries, mh_mask, None, "linear",
                            hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        )

                        x = residual_connection(x, y, is_training, keep_prob_res)

                    with tf.variable_scope("bmh_path_attention"):
                        y = multihead_attention_decoder(
                            x, bmh_overall_queries, bmh_mask, None, "linear",
                            hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        )

                        x = residual_connection(x, y, is_training, keep_prob_res)

                    x = tf.Print(x, [tf.shape(x), tf.shape(y), tf.shape(one_hop_list), tf.shape(one_hop_mask)], first_n=10)
                    print("Method called is "+method+" path_attend_5bs_all", flush=True)
                else:
                    print("no AKW component");

                with tf.variable_scope("ffn"):
                    x = residual_connection_with_dense(
                        x, 4 * hn, True, 0., "res_ffn", act_name, False,
                        wd, keep_prob_dense, is_training, keep_prob_res)


    new_decoder_history_inputs = None
    if use_decoder_history:
        cur_history_inputs = tf.stack(cur_history_inputs_list, axis=1)  # bs,num_layer,1,hn
        new_decoder_history_inputs = tf.concat([decoder_history_inputs, cur_history_inputs], axis=2)
    return x, new_decoder_history_inputs


def multihead_attention_decoder(
        tensor_from, tensor_to, mask_to, mask_direction=None,  # [bs,slf,slt]
        act_name="relu", hn=768, head_num=12, wd=0., is_training=None, keep_prob_dense=1., keep_prob_attn=1.,
        tensor_to_prev=None, mask_prev_to=None,
        scope=None,
):
    head_dim = hn // head_num
    with tf.variable_scope(scope or "multihead_attention_decoder"):
        # if not isinstance(tensor_to_prev, type(None)):  # to print the shape
        #     tensor_from = tf.Print(tensor_from, [
        #         tf.shape(tensor_from), tf.shape(tensor_to),  tf.shape(mask_to),  tf.shape(tensor_to_prev)])

        if isinstance(tensor_to_prev, type(None)):
            tensor_to_all = tensor_to # bs,sl,hn
            mask_to_all = mask_to  # bs,sl
        else:
            tensor_to_all = tf.concat([tensor_to_prev, tensor_to], -2)  # bs,psl+1,hn
            if mask_prev_to is None:
                mask_prev_to = tf.cast(tf.ones(get_shape_list(tensor_to_prev, 3)[:2] , tf.int32), tf.bool)  # bs,psl
            mask_to_all = tf.concat([mask_prev_to, mask_to], -1)  # bs,psl+1

        attn_scores = compatibility_fn(
            tensor_from, tensor_to_all, method="multi_head", head_num=head_num,
            hn=hn, wd=wd, is_training=is_training, keep_prob=keep_prob_dense,
        )  # [bs,hd_num,slf,slt]
        # logging.info("inside the decoder dense layer")
        v_heads = bn_dense_layer_v2(  # bs,slt,hd_dim * hd_num
            tensor_to_all, head_dim, True, 0., 'v_heads',
            'linear', False, wd, keep_prob_dense, is_training, dup_num=head_num
        )
        v_heads = split_head(v_heads, head_num)  # # bs,hd_num,slt,hd_dim

        # mask the self-attention scores
        attn_scores_mask = tf.expand_dims(mask_to_all, 1)  # bs,1,tsl
        if (not isinstance(mask_direction, type(None))) and isinstance(tensor_to_prev, type(None)):
            attn_scores_mask = tf.logical_and(attn_scores_mask, mask_direction)  # bs,tsl,tsl
            # attn_scores_mask here would still be equal to the masl_direction
        attn_scores_masked = exp_mask_v3(attn_scores, attn_scores_mask, multi_head=True)  # [bs,hd_num,slf,slt]
        # attn_scores_masked would have very high -ve numbers at the locations where attention should not be seen.
        
        
        attn_prob = tf.nn.softmax(attn_scores_masked)
        attn_prob = dropout(attn_prob, keep_prob_attn, is_training)  # [bs,hd_num,slf,slt]

        v_heads_etd = tf.expand_dims(v_heads, 2)  # bs,hd_num,1,slt,hd_dim
        attn_prob_etd = tf.expand_dims(attn_prob, -1)  # bs,hd_num,slf,slt,1

        attn_res = tf.reduce_sum(v_heads_etd * attn_prob_etd, 3)  # bs,hd_num,slf,hd_dim
        out_prev = combine_head(attn_res)  # bs,fsl,hn

        # if mask_direction is not None and tensor_to_prev is None:
        #     attn_scores = exp_mask_v3(attn_scores, mask_direction, multi_head=True)  # [bs,hd_num,slf,slt]
        # attn_scores = dropout(attn_scores, keep_prob_attn, is_training)
        #
        # attn_res = softsel( # [bs,hd_num,slf,dhn]
        #     v_heads, attn_scores, mask_to_all,
        #     mask_add_head_dim_for_scores=True,
        #     input_add_multi_head_dim=False,
        #     score_add_hn_dim=True,
        #     axis=3)
        # out_prev = combine_head(attn_res)
        # dense layer
        out = bn_dense_layer_v2(
            out_prev, hn, True, 0., "output_transformer", act_name, False, wd, keep_prob_dense, is_training
        )
        return out



# self-attention
def s2t_self_attn(  # compatible with lower version of tensorflow
        tensor_input, tensor_mask, deep_act=None, method='multi_dim',
        wd=0., keep_prob=1., is_training=None,
        scope=None, **kwargs
):
    use_deep = isinstance(deep_act, str)  # use Two layers or Single layer for the alignment score
    with tf.variable_scope(scope or 's2t_self_attn_{}'.format(method)):
        tensor_shape = get_shape_list(tensor_input)
        hn = tensor_shape[-1]  # hidden state number

        if method == 'additive':
            align_scores = bn_dense_layer_v2(  # bs,sl,hn/1
                tensor_input, hn if use_deep else 1, True, 0., 'align_score_1', 'linear', False,
                wd, keep_prob, is_training
            )
            if use_deep:
                align_scores = bn_dense_layer_v2(  # bs,sl,1
                    act_name2fn(deep_act)(align_scores), 1, True, 0., 'align_score_2', 'linear', False,
                    wd, keep_prob, is_training
                )
        elif method == 'multi_dim':
            align_scores = bn_dense_layer_v2(  # bs,sl,hn
                tensor_input, hn, False, 0., 'align_score_1', 'linear', False,
                wd, keep_prob, is_training
            )
            if use_deep:
                align_scores = bn_dense_layer_v2(  # bs,sl,hn
                    act_name2fn(deep_act)(align_scores), hn, True, 0., 'align_score_2', 'linear', False,
                    wd, keep_prob, is_training
                )
        elif method == 'multi_dim_head':
            get_shape_list(tensor_input, expected_rank=3)  # the input should be rank-3
            assert 'head_num' in kwargs and isinstance(kwargs['head_num'], int)
            head_num = kwargs['head_num']
            assert hn % head_num == 0
            head_dim = hn // head_num

            tensor_input_heads = split_head(tensor_input, head_num)  # [bs,hd,sl,hd_dim]

            align_scores_heads = bn_dense_layer_multi_head(  # [bs,hd,sl,hd_dim]
                tensor_input_heads, head_dim, True, 0., 'align_scores_heads_1', 'linear', False,
                wd, keep_prob, is_training
            )
            if use_deep:
                align_scores_heads = bn_dense_layer_multi_head(  # [bs,hd,sl,hd_dim]
                    act_name2fn(deep_act)(align_scores_heads), head_dim,
                    True, 0., 'align_scores_heads_2', 'linear', False,
                    wd, keep_prob, is_training
                )
            align_scores = combine_head(align_scores_heads)  # [bs,sl,dim]
        else:
            raise AttributeError

        # attention procedure align_scores [bs,sl,1/dim]
        align_scores_masked = exp_mask_v3(align_scores, tensor_mask, multi_head=False, high_dim=True)  # bs,sl,hn
        attn_prob = tf.nn.softmax(align_scores_masked, dim=len(get_shape_list(align_scores_masked))-2)  # bs,sl,hn

        if 'attn_keep_prob' in kwargs and isinstance(kwargs['attn_keep_prob'], float):
            attn_prob = dropout(attn_prob, kwargs['attn_keep_prob'], is_training)  # bs,sl,hn

        attn_res = tf.reduce_sum(  # [bs,sl,hn] -> [bs,dim]
            mask_v3(attn_prob*tensor_input, tensor_mask, high_dim=True), axis=-2
        )

        return attn_res  # [bs,hn]