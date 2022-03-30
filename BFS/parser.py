from nltk.tokenize import WordPunctTokenizer
import random
import time
import timeout_decorator
from utils.csqa import load_json, get_ent_int_id
import copy


random.seed(2018)
def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return ' '.join(sentence)

#defind base action, paper A1-A15
Base_action={
"A1":("S","set"),  #A1:S->set
"A2":("S","num"),  #A2:S->num
"A3":('S','bool'), #A3:S->bool
"A4":("set",('find',"set","r")), #A4:set->find(set,r)
"A5":("num",("count","set")),   #A5:num->count(set)
"A6":('bool',('in','e','set')), #A6:bool->in(e,set)
"A7":('set',('union','set','set')), #A7:set->union(set1,set2)
'A8':('set',('inter','set','set')), #A8:set->inter(set1,set2)
"A9":('set',('diff','set','set')),#A9:set->diff(set1,set2)
    
'A10':('set',('>','Num','Tuple_Count')), #for paper A10-A14, we use A10-14 and A26-A28 to implement them
'A11':('set',('<','Num','Tuple_Count')), #for Num, it come from num (obtain by operation) 
'A12':('set',('=','Num','Tuple_Count')), #or num_utterence (extract form utterence)
'A13':('set',('argmax','Tuple_Count')),  #for paper A10:set->larger(set,r,num), equal to A10A24(or A25)A26A27(or A28)     
'A14':('set',('argmin','Tuple_Count')),  #but only count the triple such that (entity with type1,r,entity with type2)
"A15":('set',('{}','e')), #A15:set->{e} 
 
#extra actions: 
    
#A22 help reverse relation, A23 help add isA relation to graph
#A26-A28 help implement function of paper A7-A9
"A22":("set",("find_reverse","set","r")), #S->find(set,reverse(r))
"A23":('set',('filter','Type','set')), #S->find(set,"isA Type")
    
#A24-A25 distinguish which number is instantiated
'A24':('Num','num'), #Num come from operation
'A25':('Num','num_utterence'),#Num come from untterence  
    
#A26-A28 help implement function of paper A7-A9    
'A26':('Tuple_Count',('traverse','Count','Tuple')), 
"A27":('Tuple',('Pre_Type','r','Type','Type')),
"A28":('Tuple',('Reverse_pre_Type','r','Type','Type')),
}


def prune(action_sequences,action):  # todo: figure me out
    #This function mainly defind some prune rules to avoid dead loop in BFS search
    need_prune=False
    #these actions only use at most once
    conflict_action=["A7","A8",'A9']
    if action in conflict_action and len(set(conflict_action)&set(action_sequences))!=0:
        need_prune=True
        
    #these actions also only use at most once
    conflict_action=['A10','A11','A12','A13','A14']
    if action in conflict_action and len(set(conflict_action)&set(action_sequences))!=0:
        need_prune=True
        
    #these actions can't apply itself, for example, it's easy to search A23A23A23....A23 so that it leads to dead loop
    conflict_action=['A23']
    if len(action_sequences)!=0 and action in conflict_action and action==action_sequences[-1]:
        need_prune=True
    
        
    #A15 can't follow these actions
    conflict_action=['A1','A5','A6','A7','A8','A9','A23']
    if action=="A15" and len(action_sequences)!=0 and action_sequences[-1] in conflict_action:
        need_prune=True
        
    #without copy, A15 must follow A4,A22
    conflict_action=['A4','A22']
    if len(action_sequences)!=0 and action_sequences[-1] in conflict_action and action!="A15":
        need_prune=True
    
    #these actions only use twice continuously, thus at most 2-hop and prevent dead loop
    conflict_action=['A4','A22']
    if len(action_sequences)>1 and action_sequences[-2] in conflict_action and action_sequences[-1] in conflict_action and action in conflict_action:
        need_prune=True   
    
    #all action occurs at most three times
    cont=0
    for x in action_sequences:
        if x==action:
            cont+=1
    if cont>=3:
        need_prune=True 
    return need_prune


class Parser(object):
    def __init__(self,database,use_op_type_constraint=False):
        # child2parent code
        self.child_id2parent = None
        self.database=database
        self.use_op_type_constraint = use_op_type_constraint
        random.seed(2018)
        self.load_child2parent()

    def load_child2parent(self):  # xxx added
        if self.child_id2parent is None and self.use_op_type_constraint:
            self.child_id2parent = load_json("data/kb/child_par_list.json")

    def from_same_type(self, s1, s2): # xxx added
        if not self.use_op_type_constraint:
            return True
        if self.child_id2parent is None:
            self.load_child2parent()

        s1_list = list(s1)
        s2_list = list(s2)

        max_ent_num = max(len(s1_list), len(s2_list))
        s1_types, s2_types = set(), set()
        for _idx_e in range(max_ent_num):

            if _idx_e < len(s1_list):
                if not (isinstance(s1_list[_idx_e], str) and s1_list[_idx_e].startswith("Q")):
                    return False
                else:
                    type_code = self.child_id2parent[get_ent_int_id(s1_list[_idx_e])]
                    if type_code is None:
                        return False
                    s1_types.add(type_code)
            if _idx_e < len(s2_list):
                if not (isinstance(s2_list[_idx_e], str) and s2_list[_idx_e].startswith("Q")):
                    return False
                else:
                    type_code = self.child_id2parent[get_ent_int_id(s2_list[_idx_e])]
                    if type_code is None:
                        return False
                    s2_types.add(type_code)
            if len(s1_types.intersection(s2_types)) > 0:
                return True
        return False

        # # Method 1
        # s1_types = set(self.child_id2parent[get_ent_int_id(_elem)] for _elem in s1)
        # s2_types = set(self.child_id2parent[get_ent_int_id(_elem)] for _elem in s2)
        # if len(s1_types.intersection(s2_types)) > 0:
        #     return True
        # else:
        #     return False
        # # Method 0
        # set_type = None
        # if len(s1) > 0 and len(s2) > 0:
        #     for elem in set(s1).union(set(s2)):
        #         elem_type = self.child2parent.get(elem)
        #         if elem_type is not None:
        #             set_type = set_type or elem_type
        #             if set_type != elem_type:
        #                 return False
        # return True



    def parsing_answer(self,all_entities,utterance,question_type):  # read: check whether the type is bool or number
        #extract answers from utterances.
        bool_answer=[]
        cont_answer=[]
        sentence=tokenize(utterance).split()
        for x in sentence:
            if x=='yes' or x=='no':
                bool_answer.append(x)
            try:
                cont_answer.append(int(x))
            except:
                continue
        if 'Bool' in question_type and len(bool_answer)!=0:
            return bool_answer
        if 'Count' in question_type and len(cont_answer)!=0:
            return cont_answer
        return set(all_entities)     

    def build_action(self,entities,predicates,types,numbers):  # read: load relation (r), entity (e), Type (t) and 'num_utterence' to action dictionary
        #defined actions for instantiation, paper A16-A18
        base_len=1000  # use base 1000 to store non-initiated
        action=Base_action.copy()  # read: pre-defined actions
        for p in predicates:
            action['A'+str(base_len+1)]=("r",p)
            base_len+=1
        for e in entities:
            action['A'+str(base_len+1)]=("e",e)
            base_len+=1
        for t in types:
            action['A'+str(base_len+1)]=("Type",t)
            base_len+=1   
        for n in numbers:
            action['A'+str(base_len+1)]=('num_utterence',n)
            base_len+=1           

        return action

    @staticmethod
    def build_state_action(Action): #  todo # read: action constrain: action_state (e.g., set, S, num) -> action_list (e.g., A2, ..)
        """build a dictionary
        key: semantic category
        values: list of legal actions
        """
        state_action={}
        for x in Action:
            try:
                state_action[Action[x][0]].append(x)
            except:
                state_action[Action[x][0]]=[x]
                
        return state_action 
    
    def op(self,op,arguments):  # todo: read me
        #implement function of operation

        #This is for unaryop such that only one argument
        if len(arguments)==1:
            x=arguments[0]
            if op=="count":  # return a length of the input set
                if type(x)!=set:
                    return None
                else:
                    return len(x)
            elif op=="argmax":  # return a set whose elements are largest
                if type(x)==list and len(x)!=0 and type(x[0][1])==int:
                    x.sort(key=lambda x:-x[1])
                    max_cont=x[0][1]
                    res=set()
                    for item in x:
                        if item[1]==max_cont:
                            res.add(item[0])
                    return res
                else:
                    return None
            elif op=="argmin":  # return a set whose elements are smallest
                if type(x)==list and len(x)!=0 and type(x[0][1])==int:
                    x.sort(key=lambda x:x[1])
                    min_cont=x[0][1]
                    res=set()
                    for item in x:
                        if item[1]==min_cont:
                            res.add(item[0])
                    return res
                else:
                    return None
            elif op=='{}':
                if type(x)==str:
                    return set([x])
                else:
                    return None
            else:
                return None
            
        #This is for biop such that only two argument
        if len(arguments)==2:       
            x,y=arguments[0],arguments[1]
            if op=='in':  # read
                if type(y)!=set:
                    return None
                else:
                    if x in y:
                        return ['yes']
                    else:
                        return ['no']
            elif op=='inter':  # read
                if type(x)==set and type(y)==set:
                    if not self.from_same_type(x, y):  # xxx added
                        return None
                    temp=x&y
                    if len(temp)==0:
                        return None
                    return temp
                else:
                    return None
            elif op=='union':
                if type(x)!=set or type(y)!=set:
                    return None
                else:
                    if not self.from_same_type(x, y):  # xxx added
                        return None
                    temp=x|y
                    if len(temp)==0:
                        return None
                    return temp
            elif op=='diff':
                if type(x)!=set or type(y)!=set:
                    return None
                else:
                    if not self.from_same_type(x, y):  # xxxx added
                        return None
                    temp=x-y
                    if len(temp)==0:
                        return None
                    return temp
            elif op=='find':
                if type(x)==set and type(y)==str and y.startswith('P'):
                    temp=set()
                    for _x in x:
                        if type(_x)==str and _x.startswith('Q'):
                            t=self.database.sub_pre(_x,y)
                            if t is None:
                                return None
                            else:
                                temp=temp|t
                   
                        else:
                            return None
                    if len(temp)==0:
                        return None
                    else:
                        return temp
                else:
                    return None
            elif op=='find_reverse':  # read
                if type(x)==set and type(y)==str and y.startswith('P'):
                    temp=set()
                    for _x in x:
                        if type(_x)==str and _x.startswith('Q'):
                            t=self.database.obj_pre(_x,y)
                            if t is None:
                                return None
                            else:
                                temp=temp|t
                   
                        else:
                            return None
                    if len(temp)==0:
                        return None
                    else:
                        return temp
                else:
                    return None
            elif op=="filter":  # read
                if type(x)==str and type(y)==set and x.startswith('Q'):
                    answer=[]
                    for item in y.copy():
                        if item.startswith('Q') and self.database.entity_type(item)==x:
                            answer.append(item)
                    answer=set(answer)
                    if len(answer)==0 or answer==y:
                        return None
                    else:
                        return answer                      
                else:
                    return None
            elif op=='traverse':
                if x=="Count" and type(y)==list and len(y)!=0 and type(y[0][1])==set:
                        res=[(item[0],len(item[1])) for item in y]
                        return res
                else:
                    return None
            elif op=='>':
                if type(x)==int and type(y)==list and len(y)!=0 and type(y[0][1])==int:
                    res=set()
                    for item in y:
                        if item[1]>x:
                            res.add(item[0])
                    if len(res)==0:
                        return None
                    else:
                        return res
                else:
                    return None
            elif op=='=':
                if type(x)==int and type(y)==list and len(y)!=0 and type(y[0][1])==int:
                    res=set()
                    for item in y:
                        if item[1]==x:
                            res.add(item[0])
                    if len(res)==0:
                        return None
                    else:
                        return res
                else:
                    return None  
            elif op=='<':
                if type(x)==int and type(y)==list and len(y)!=0 and type(y[0][1])==int:
                    res=set()
                    for item in y:
                        if item[1]<x:
                            res.add(item[0])
                    if len(res)==0:
                        return None
                    else:
                        return res
                else:
                    return None
            elif op=='>=':
                if type(x)==int and type(y)==list and len(y)!=0 and type(y[0][1])==int:
                    res=set()
                    for item in y:
                        if item[1]>=x:
                            res.add(item[0])
                    if len(res)==0:
                        return None
                    else:
                        return res
                else:
                    return None
            elif op=='<=':
                if type(x)==int and type(y)==list and len(y)!=0 and type(y[0][1])==int:
                    res=set()
                    for item in y:
                        if item[1]<=x:
                            res.add(item[0])
                    if len(res)==0:
                        return None
                    else:
                        return res
                else:
                    return None

            else:
                return None
            
        #This is for trop such that only three argument
        if len(arguments)==3:
            x,y,z=arguments[0],arguments[1],arguments[2]
            if op=='Pre_Type':
                if type(x)==str and type(y)==str and type(z)==str and x.startswith('P') and y.startswith('Q') and z.startswith('Q'): 
                    temp=self.database.pre_sub_type(y,x,z)
                    if temp is None or len(temp)==0:
                        return None                  
                    return temp
                else:
                    return None
            elif op=='Reverse_pre_Type':
                if type(x)==str and type(y)==str and type(z)==str and x.startswith('P') and y.startswith('Q') and z.startswith('Q'): 
                    temp=self.database.pre_obj_type(y,x,z)

                    if temp is None or len(temp)==0:
                        return None                
                    return temp
                else:
                    return None
            else:
                return None   
         
    def execute_lf(self,lf,history):
        ##execute logical form to get answer, also keep (logical form, answer) pair in history
        if type(lf)!=tuple:
            return None
        if lf in history:
            return history[lf]
        items=[]
        for x in lf:
            if x is None:
                return None
            elif type(x)==tuple:
                items.append(self.execute_lf(x,history))
            else:
                items.append(x)
        if len(items)<=4:
            history[lf]=self.op(items[0],items[1:])
            return history[lf]
        else:
            return None
        
    @timeout_decorator.timeout(20)    # un-comment me for regular use
    def BFS(self,entities,pres,types,numbers,beam_size):

        self.beamsize=beam_size

        action=self.build_action(entities,pres,types,numbers)
        state_action=Parser.build_state_action(action)
        

        # read: core code
        history_menory={}  #
        que=[['S']]  # store the constrain
        stack=[[]]  # this is the action/operator
        que_depth=[[0]]  #
        stack_depth=[[]]  #
        stack_answer=[[]]  #
        Action=[[]]  #
        search_depth=[0]  #
        abstract_forms=[[]]  #
        # entity_record = [{"e_leaf": set(), "e_non_leaf": set(), "t": set(), "r": set()}]  # added: for entity record
        logical_forms=[]  #
        answers=[]  #å
        logical_action=[]  #
        # lf_entity_record = []
        depth=0  #
        while len(que)!=0:
            if depth!=search_depth[0]:  # when move to next level, random drop some branch and shuffle all
                depth=search_depth[0]
                index=list(range(len(que)))
                random.shuffle(index)
                index=index[:self.beamsize-len(answers)]  # random drop
                que=[que[i] for i in index]
                stack=[stack[i] for i in index]
                que_depth=[que_depth[i] for i in index]
                stack_depth=[stack_depth[i] for i in index]
                stack_answer=[stack_answer[i] for i in index]
                search_depth=[search_depth[i] for i in index]
                Action=[Action[i] for i in index]
                abstract_forms=[abstract_forms[i] for i in index]
                # entity_record = [entity_record[i] for i in index]
            q=que.pop(0)  # choose a brach, begin to expand, FIFO for bfs
            s=stack.pop(0)
            q_d=que_depth.pop(0)
            s_d=stack_depth.pop(0)   
            s_a=stack_answer.pop(0)
            d=search_depth.pop(0)
            a=Action.pop(0)
            a_f=abstract_forms.pop(0)
            # e_r = entity_record.pop(0)  # added: for entity record

            if len(q)==0:  # if queue is empty, it means the logical form is end insert logical form
                if len(s)!=0:  #
                    logical_forms.append(s[0])
                    answers.append(s_a[0])
                    logical_action.append(a_f)
                    # lf_entity_record.append(e_r)
                    #print(a)
                    if len(answers)>=self.beamsize:  # may cause by beamsize
                        break
                continue
                
            #if q[0] is terminating symbol,insert stack. Otherwise, excute action to state.
            if type(q[0])==str and q[0] in state_action:  # the q[0] is a type -> substitute the "type" to "action+args"
                if len(a)>0:
                    last_action=a[-1]
                else:
                    last_action=""
                legitimate_actions = state_action[q[0]]
                for x in legitimate_actions:  #
                    if prune(a,x):  # a is the history action and the x is the new coming action
                        continue
                    # open a new branch for +x
                    s_temp=s.copy()
                    s_d_temp=s_d.copy()
                    q_temp=q.copy()
                    q_d_temp=q_d.copy()
                    s_a_temp=s_a.copy()
                    a_temp=a.copy()
                    a_f_temp=a_f.copy()
                    # e_r_temp = copy.deepcopy(e_r)  # added: for entity record
                    next_state=action[x][1]
                    if type(next_state)!=tuple:  # here the next state is not an action, this mean no branch only for type extension
                        q_temp=[next_state]+q[1:]  # the required arg is one, just substitute
                        que.append(q_temp)
                        stack.append(s_temp)  # new branch
                        stack_depth.append(s_d_temp)  # new branch
                        que_depth.append(q_d_temp)  # new branch, q_d not change
                        stack_answer.append(s_a_temp)
                        search_depth.append(d+1) # for this bew branch, search depth +1
                        if int(x[1:])<1000:
                            a_temp.append(x)
                            a_f_temp.append(('Action',x))
                        else:  # for entity, type and predicates
                            a_f_temp.append((q[0],next_state))
                            # if q[0] == 'e':  # added
                            #     e_r_temp['e_leaf'].add(next_state)  # added
                            # elif q[0] == 'r':
                            #     e_r_temp["r"].add(next_state)
                            # elif q[0] == 'Type':
                            #     e_r_temp["t"].add(next_state)
                        Action.append(a_temp)  # for new branch
                        abstract_forms.append(a_f_temp)  # because it has been executed w/o KB support
                        # entity_record.append(e_r_temp)  # added
                    else:   # here the next state is an action
                        q_temp=list(next_state)+q[1:]  # use new action and action to substitute the type_constrain
                        q_d_temp=[q_d_temp[0]+1 for i in range(len(next_state))]+q_d_temp[1:]  # ** increase q depth
                        que.append(q_temp)
                        stack.append(s_temp)
                        stack_depth.append(s_d_temp)
                        que_depth.append(q_d_temp)
                        stack_answer.append(s_a_temp)
                        search_depth.append(d+1)
                        if int(x[1:])<1000:
                            a_temp.append(x)
                            a_f_temp.append(('Action',x))
                        else:
                            a_f_temp.append((q[0],next_state))
                            # if q[0] == 'e':  # added
                            #     e_r_temp["e_leaf"].add(next_state)  # added
                            # elif q[0] == 'r':
                            #     e_r_temp["r"].add(next_state)
                            # elif q[0] == 'Type':
                            #     e_r_temp["t"].add(next_state)
                        Action.append(a_temp)
                        abstract_forms.append(a_f_temp)  # because it has been executed w/o KB support
                        # entity_record.append(e_r_temp)  # added
            else:  # if 1st in queue is a action rather than type expansion
                s_temp=s.copy()
                s_d_temp=s_d.copy()
                q_temp=q.copy()
                q_d_temp=q_d.copy()
                s_a_temp=s_a.copy()
                a_temp=a.copy()
                a_f_temp=a_f.copy()
                # e_r_temp = copy.deepcopy(e_r)  # added: for entity record
                s_a_temp.append(q[0])  # !!! push "action name" to stack answer
                s_temp.append(q[0])  # !!!  push "action name" to stack
                s_d_temp.append(q_d[0])  # !!! push the layer of action to stack depth
                q_temp=q[1:]
                q_d_temp=q_d[1:]              
                flag=True  # figure me out to run the tree: Cond1)over   Cond2) stack's
                while (len(q_d_temp)==0 and len(s_temp)>1) or (len(q_d_temp)!=0 and s_d_temp[-1]>q_d_temp[0]):  # todo: read this block
                    current_depth=s_d_temp[-1]
                    current_s=[]
                    current_s_a=[]
                    while len(s_d_temp)>0 and s_d_temp[-1]==current_depth:  # pop all in corresponding layer from stack
                        current_s.insert(0,s_temp.pop(-1))
                        current_s_a.insert(0,s_a_temp.pop(-1))
                        s_d_temp.pop(-1)
                    if len(current_s)==3 and current_s[1]==current_s[2]:
                        flag=False
                        break
                    if tuple(current_s) in history_menory:
                        local_answer=history_menory[tuple(current_s)]
                    else:
                        local_answer=self.op(current_s_a[0],current_s_a[1:])
                        history_menory[tuple(current_s)]=local_answer
                    if local_answer is None:
                        flag=False
                        break
                                        
                    s_a_temp.append(local_answer)
                    s_d_temp.append(current_depth-1)
                    s_temp.append(tuple(current_s))
                    # if isinstance(local_answer, set) and current_s_a[0]!='{}':
                    #     e_r_temp["e_non_leaf"].update(local_answer)
                if flag:  # only valid run result can be added to the queue
                    que.append(q_temp)
                    stack.append(s_temp)
                    stack_depth.append(s_d_temp)
                    que_depth.append(q_d_temp)   
                    stack_answer.append(s_a_temp)
                    search_depth.append(d+1)
                    Action.append(a_temp)
                    abstract_forms.append(a_f_temp)
                    # entity_record.append(e_r_temp)
                    
        return logical_forms,answers,logical_action,{}
  
