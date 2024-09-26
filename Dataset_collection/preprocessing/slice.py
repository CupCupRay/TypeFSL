import os
import re
import json
import time
import signal
import binaryninja
import numpy as np
import networkx as nx
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def _handle_timeout(signum, frame):
    raise TimeoutError('Timeout')


def asm_split_inst(ins, symbol_map, string_map):
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    if len(parts) > 1:
        operand = parts[1:]
    for i in range(len(operand)):
        symbols = re.split('([0-9A-Za-z]+)', operand[i])
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) >= 6:
                if int(symbols[j], 16) in symbol_map:
                    symbols[j] = "symbol" # function names
                elif int(symbols[j], 16) in string_map:
                    symbols[j] = "string" # constant strings
                else:
                    symbols[j] = "address" # addresses
        operand[i] = ' '.join(symbols)
    opcode = parts[0]
    return ' '.join([opcode]+operand)


def il_split_inst(ins, symbol_map, string_map):
    # print('Hanlding: ', ins)
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    for i in range(len(parts)):
        symbols = re.split('([0-9A-Za-z_]+)', parts[i])
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) >= 6:
                if int(symbols[j], 16) in symbol_map:
                    symbols[j] = "symbol" # function names
                elif int(symbols[j], 16) in string_map:
                    symbols[j] = "string" # constant strings
                else:
                    symbols[j] = "address" # addresses
        parts[i] = ' '.join(symbols)
    return ' '.join(parts)


def collect_df_sequence(g:nx.DiGraph, var_addrs, symbol_map, string_map, mode, slice_mode='fb'):
    sequence = list()
    for node in g:
        # print(node, g._node[node])
        if node == -1 or 'text' not in g._node[node] or g._node[node]['text'] == None:
            continue
        if node in var_addrs:
            if mode == 'asm': 
                sequence.append([hex(node), 
                                 il_split_inst(g._node[node]['il_text'], symbol_map, string_map),
                                 asm_split_inst(g._node[node]['text'], symbol_map, string_map)])
            else: 
                sequence.append((hex(node), il_split_inst(g._node[node]['text'], symbol_map, string_map)))
           
            if 'f' in slice_mode:
                # Forward 
                taken_nbs, cur_nodes = [], [node]
                while True:
                    # Obtain the next nodes
                    init_next_nbs = []
                    for each_cur in cur_nodes:
                        init_next_nbs += list(g.successors(each_cur)) 
                    
                    # Check duplication and break condition
                    next_nbs = []
                    for nbs in init_next_nbs:
                        if nbs not in taken_nbs:
                            next_nbs.append(nbs)
                    if len(next_nbs) == 0: 
                        break
                    next_nbs = list(set(next_nbs))
                    
                    # Obtain the contents of next nodes
                    for next_n in next_nbs:
                        if next_n == -1 or 'text' not in g._node[next_n] or g._node[next_n]['text'] == None:
                            continue
                        if mode == 'asm': 
                            sequence.append([hex(next_n), 
                                             il_split_inst(g._node[next_n]['il_text'], symbol_map, string_map),
                                             asm_split_inst(g._node[next_n]['text'], symbol_map, string_map)])
                        else: 
                            sequence.append((hex(next_n), il_split_inst(g._node[next_n]['text'], symbol_map, string_map)))
        
                    # Set next loop
                    for nbs in next_nbs:
                        taken_nbs.append(nbs)
                    cur_nodes = next_nbs
               
            if 'b' in slice_mode: 
                # Backward 
                taken_nbs, cur_nodes = [], [node]
                while True:
                    # Obtain the previous nodes
                    init_prev_nbs = []
                    for each_cur in cur_nodes:
                        init_prev_nbs += list(g.predecessors(each_cur)) 
                    
                    # Check duplication and break condition
                    prev_nbs = []
                    for nbs in init_prev_nbs:
                        if nbs not in taken_nbs:
                            prev_nbs.append(nbs)
                    if len(prev_nbs) == 0: 
                        break
                    prev_nbs = list(set(prev_nbs))
                    
                    # Obtain the contents of previous nodes
                    for prev_n in prev_nbs:
                        if prev_n == -1 or 'text' not in g._node[prev_n] or g._node[prev_n]['text'] == None:
                            continue
                        if mode == 'asm': 
                            sequence.append([hex(prev_n), 
                                             il_split_inst(g._node[prev_n]['il_text'], symbol_map, string_map),
                                             asm_split_inst(g._node[prev_n]['text'], symbol_map, string_map)])
                        else: 
                            sequence.append((hex(prev_n), il_split_inst(g._node[prev_n]['text'], symbol_map, string_map)))
        
                    # Set next loop
                    for nbs in prev_nbs:
                        taken_nbs.append(nbs)
                    cur_nodes = prev_nbs
    
    # Deduplication and sort by address 
    # sequence = list(set(sequence))
    temp, new_sequence = list(), list()
    for s in sequence:
        if s[0] not in temp:
            temp.append(s[0])
            new_sequence.append(s)
    result = sorted(new_sequence, key=lambda x:(int(x[0], 16)))
    return result


class ArgumentSlicer:

    def __init__(self, bv:binaryninja.BinaryView, func:binaryninja.Function, mode):
        self.bv = bv
        self.func = func
        self.mode = mode
        self.G = nx.DiGraph()
        self.G.add_node(-1, text='entry_point')
        self.label_dict = dict()
        self.variable_dict = dict()
        self.label_dict[-1] = 'entry_point'
        
    def do_slice(self, func_num, total_time):
        # Constraint the handling time
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(100)
        
        try:
            T1 = time.perf_counter()
            if self.mode == 'hlil':
                this_func = self.func.hlil
            else: this_func = self.func.mlil
            
            # Collect the var source
            for var in self.func.vars:
                source_addr = list()
                if not re.match(r"arg[\d+]", var.name):
                    # if not self.func.name.startswith('__'):
                        # print('WARNING: No Strip!', self.func.name, var.name)
                    continue
                for ref in self.func.get_mlil_var_refs(var):
                    source_addr.append(ref.address)
                self.variable_dict[var.name] = source_addr
            
            callee_ret = list()
            
            for block in this_func:
                for ins in block:
                    insn_text = None
                    if self.mode == 'asm':
                        insn_text = self.bv.get_disassembly(ins.address)
                        self.G.add_node(ins.address, text=insn_text)
                        self.label_dict[ins.address] = insn_text

                        il_text = str(ins)
                        self.G.add_node(ins.address, il_text=il_text)

                    else:
                        insn_text = str(ins)
                        self.G.add_node(ins.address, text=insn_text)
                        self.label_dict[ins.address] = insn_text

                    
                    
                    if insn_text is None:
                        print('WARNING: No instruction in', ins.address)
                        continue
                    
                    # Do not trace the return value data-flow for callee
                    if ('arm' in str(self.bv.arch) and 'bl' in insn_text) or \
                    ('x86' in str(self.bv.arch) and 'call' in insn_text) or \
                    ('mips' in str(self.bv.arch) and 'jal' in insn_text):
                        for var in ins.vars_written:
                            if var not in callee_ret:
                                callee_ret.append(var)
                            elif var in callee_ret:
                                callee_ret.remove(var)
                    
                    depd = []
                    
                    # Backward
                    for var in ins.vars_read:
                        # Do not trace data-flow of the callee's ret
                        if var in callee_ret:
                            continue
                        
                        """
                        if var.name not in self.variable_dict:
                            self.variable_dict[var.name] = ins.address
                            """
                            
                        depd += [(definition.address, ins.address)
                                 for definition in this_func.get_var_definitions(var)
                                 if definition.address != ins.address]
                    
                    # Forward 
                    for var in ins.vars_written:
                        # Do not trace data-flow of the callee's ret
                        if var in callee_ret:
                            continue
                        
                        """
                        if var.name not in self.variable_dict:
                            self.variable_dict[var.name] = [ins.address]
                        else:
                            temp_addr = self.variable_dict[var.name]
                            temp_addr.append(ins.address)
                            self.variable_dict[var.name] = temp_addr
                            """
                            
                        depd += [(ins.address, use.address)
                                 for use in this_func.get_var_uses(var)
                                 if use.address != ins.address]
                           
                    if depd:
                        self.G.add_edges_from(depd)
            T2 = time.perf_counter()
            func_num += 1
            total_time += (T2 - T1)
        except Exception as e:
            print(e)
        finally:
            signal.alarm(0)
        
        return self.G, func_num, total_time
    
    def get_label_dict(self):
        return self.label_dict
    
    def get_variable_dict(self):
        return self.variable_dict
    

def find_callee_dataflow(bv:binaryninja.BinaryView, main_var_insn, arg_insns, func_addr, callee_insn):
    callee_dataflow = list()
    for func in bv.functions:
        # Collect the callee information
        call_insn = None
        total_insns = list()
        try:
            if func.address_ranges[0].start == func_addr:
                # print('Start to find callee dataflow of', hex(func_addr))
                for block in func.mlil:
                    for insn in block:
                        if insn.address == int(callee_insn[0], 16):
                            call_insn = insn
                            break
                        total_insns.append(insn)
        except Exception as e:
            # print('WARNING: Function', func.name, 'cannot be lift into IL.')
            continue
        
        # Find the argument index for callee
        callee_addr = None
        arg_index = -1
        if call_insn is not None and call_insn:
            try:
                if ' then ' in str(call_insn) or ' else ' in str(call_insn):
                    raise Exception('Not call instruction: ' + callee_insn[0])
                call_text = str(call_insn).split('=')[-1].strip()
                callee_addr = call_text[:call_text.find('(')]
                # Deal with direct call 
                if callee_addr.startswith('0x'):
                    callee_addr = int(callee_addr, 16)
                argv = call_text[call_text.find('(') + 1: call_text.find(')')].split(', ')
            except Exception as e:
                # print(e)
                # print('WARNING: Not call insn', call_insn, call_insn.vars_read, call_insn.vars_written, callee_insn)
                return callee_dataflow

            for arg_insn in arg_insns[::-1]:
                if arg_index != -1:
                    break
                for insn in total_insns:
                    if insn.address == int(arg_insn[0], 16):
                        if arg_index != -1:
                            break
                        for var in insn.vars_written:
                            if arg_index != -1 and var.name in argv and argv.index(var.name) != -1 and arg_index != argv.index(var.name):
                                print('WARNING: Mixed variables. Var_read:', insn.vars_read, '. Var_written:', insn.vars_written, '. Addr:', arg_insn[0])
                                arg_index = -2
                                break
                            if var.name in argv:
                                arg_index = argv.index(var.name) + 1
                        
        
        # Get the data-flow of the callee argument
        if arg_index > 0 and callee_addr is not None and type(callee_addr) == int:
            try:
                for callee_func in bv.functions:
                    # print(callee_func.address_ranges[0].start, callee_addr, callee_func.name)
                    if callee_func.address_ranges[0].start == callee_addr and callee_func.name in main_var_insn:
                        # print("Find", callee_func.name)
                        if 'arg' + str(arg_index) in main_var_insn[callee_func.name]:
                            # print('Successfully find callee', callee_func.name, 'arg' + str(arg_index))
                            callee_dataflow = main_var_insn[callee_func.name]['arg' + str(arg_index)]
            except Exception as e:
                # print(e)
                return callee_dataflow
    
    return callee_dataflow
    

def try_program_slicing(result_path, folder_level_path, f, mode='asm', policy='normal'):
    symbol_map = {}
    string_map = {} 
    if mode == 'mlil': print('DFG Processing (mlil code):', f)
    elif mode == 'hlil': print('DFG Processing (hlil code):', f)
    else: print('DFG Processing (asm insn):', f)
    bv = binaryninja.load(f)

    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    for string in bv.get_strings():
        string_map[string.start] = string.value

    function_graphs = {}
    function_vars = {}
    function_addr_dict = {}
    func_num, total_time = 0, 0

    # Analyze the data flow (with arguments)
    for func in bv.functions:
        # print('Function name:', func.name)
        
        func_slice = ArgumentSlicer(bv, func, mode)
        G, func_num, total_time = func_slice.do_slice(func_num, total_time)

        for node in G.nodes:
            if not G.in_degree(node):
                G.add_edge(-1, node)
        if len(G.nodes) > 2:
            function_graphs[func.name] = G
            # Append signature of function address
            func_addr = func.address_ranges[0].start
            if type(func_addr) == str and func_addr.startswith('0x'):
                func_addr = int(func_addr, 16)
            function_addr_dict[func.name] = func_addr
            function_vars[func.name] = func_slice.get_variable_dict()

    # Collect the data flow 
    function_var_inst = dict()
    for name, graph in function_graphs.items():
        # print('Function name:', name)
        this_var_inst = dict()
        this_variable_dict = function_vars[name]
        for var in this_variable_dict:
            var_name, var_addrs = var, this_variable_dict[var]
            # Only collect the argument
            if not re.match(r"arg[\d+]", var_name):
                continue
            
            sequence = collect_df_sequence(graph, var_addrs, symbol_map, string_map, mode, 'fb')
            if sequence:
                this_var_inst[var_name] = sequence
                # print(var_name + ': ' + str(sequence))
        # Append signature of function address
        if name in function_addr_dict:
            this_var_inst['func_addr'] = function_addr_dict[name]
        function_var_inst[name] = this_var_inst
    
    
    # For inter-procedure infomation
    if policy == 'inter':
        curr_function_var_inst = function_var_inst
        function_var_inst = dict()

        for name in curr_function_var_inst:
            
            this_var_inst = dict()
            curr_var_inst = curr_function_var_inst[name]
            
            # Collect each arguments
            curr_func_addr = curr_var_inst['func_addr']
            for var in curr_var_inst:
                if not re.match(r"arg[\d+]", var): 
                    continue
                
                # arg_index = int(re.findall("\d+", var)[0])
                curr_sequence = curr_var_inst[var] # list()
                sequence, arg_insn_stack = list(), list()
                for insn in curr_sequence:
                    # Find the callee
                    if ('arm' in str(bv.arch) and 'bl' in insn[-1]) or \
                    ('x86' in str(bv.arch) and 'call' in insn[-1]) or \
                    ('mips' in str(bv.arch) and 'jal' in insn[-1]):
                        callee_seq = find_callee_dataflow(bv, curr_function_var_inst, arg_insn_stack, curr_func_addr, insn)
                        sequence = sequence + callee_seq
                    else:
                        arg_insn_stack.append(insn)
                        sequence.append(insn)
                this_var_inst[var] = sequence
            
            # Append signature of function address
            if name in function_addr_dict:
                this_var_inst['func_addr'] = function_addr_dict[name]
            function_var_inst[name] = this_var_inst
    
    # Record the results
    if mode == 'mlil': output_file = '/mlil_'
    elif mode == 'hlil': output_file = '/hlil_'
    else: output_file = '/asm_'
    
    if not os.path.exists(result_path + folder_level_path + '/'):
        os.makedirs(result_path + folder_level_path + '/')
    
    # Time 
    if func_num == 0: func_num = 1
    time_effi = {"Ave_time(ms)": (total_time / func_num) * 1000, "Func_num": func_num, "Total_time(s)": total_time}
    print(time_effi)

    with open(result_path + folder_level_path + output_file + 'var_slice_' + f.split('/')[-1] + '.json', 'w') as var_slices:
        # json.dump(function_var_inst, var_slices)
        json.dump(function_var_inst, var_slices, indent=4)
    return time_effi

