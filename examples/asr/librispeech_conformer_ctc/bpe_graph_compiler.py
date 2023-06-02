# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from typing import List, Union
from collections import defaultdict

import k2
import sentencepiece as spm
import torch


class BpeCtcTrainingGraphCompiler(object):
    def __init__(
        self,
        bpe_model_path: Path,
        device: Union[str, torch.device] = "cpu",
        topo_type = "ctc",
    ) -> None:
        """
        Args:
          lang_dir:
            This directory is expected to contain the following files:

                - bpe.model
                - words.txt
          device:
            It indicates CPU or CUDA.
          sos_token:
            The word piece that represents sos.
          eos_token:
            The word piece that represents eos.
        """
        # lang_dir = Path(lang_dir)
        # model_file = lang_dir / "bpe.model"
        sp = spm.SentencePieceProcessor()
        sp.load(str(bpe_model_path))
        self.sp = sp
        # self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
        self.device = device

        self.start_tokens = {token_id for token_id in range(sp.vocab_size()) if sp.id_to_piece(token_id).startswith("▁")}
        self.remove_intra_word_blk_flag = True
        print(f"self.remove_intra_word_blk_flag={self.remove_intra_word_blk_flag}")

        if topo_type == "hmm":
            self.max_token_id = sp.vocab_size() + 1  # hard-coded for torch audio
            self.topo = BpeCtcTrainingGraphCompiler.hmm_topo(self.max_token_id, self.start_tokens, sil_id=0)
        else:
            self.max_token_id = None
            self.topo = None

        self.topo_type = topo_type

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of piece IDs.

        Args:
          texts:
            It is a list of strings. Each string consists of space(s)
            separated words. An example containing two strings is given below:

                ['HELLO ICEFALL', 'HELLO k2']
        Returns:
          Return a list-of-list of piece IDs.
        """
        return self.sp.encode(texts, out_type=int)

    def _remove_intra_word_blk(self, decoding_graph, start_tokens, flag=True):
        c_str = k2.to_str_simple(decoding_graph)
        # print(c_str)

        arcs = c_str.split("\n")
        arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
        final_state = int(arcs[-1])
        arcs = arcs[:-1]
        arcs = [tuple(map(int, a.split())) for a in arcs]
        # print(arcs)
        # print(final_state)

        if flag is False:
            new_arcs = arcs
            new_arcs.append([final_state])

            new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
            new_arcs = [[str(i) for i in arc] for arc in new_arcs]
            new_arcs = [" ".join(arc) for arc in new_arcs]
            new_arcs = "\n".join(new_arcs)

            fst = k2.Fsa.from_str(new_arcs, acceptor=False)
            return fst

        state_arcs = defaultdict(list)
        for arc in arcs:
            state_arcs[arc[0]].append(arc)

        new_arcs = []
        for state, arc_list in state_arcs.items():
            condition1 = False
            condition2 = False
            eps_arc_i = None
            for i, arc in enumerate(arc_list):
                if arc[0] == arc[1] and arc[2] > 0:
                    condition1 = True  # We should process this kind of state
                elif arc[0] != arc[1] and arc[2] > 0 and arc[2] not in start_tokens:
                    condition2 = True
                elif arc[0] != arc[1] and arc[2] == 0:
                    eps_arc_i = i
            
            if condition1 and condition2:
                # print(f"state {state} should remove an arc {eps_self_loop}: {arc_list[eps_self_loop]}")
                new_arcs.extend(arc_list[:eps_arc_i])
                new_arcs.extend(arc_list[eps_arc_i+1:])
            else:
                new_arcs.extend(arc_list)
        new_arcs.append([final_state])

        new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
        new_arcs = [[str(i) for i in arc] for arc in new_arcs]
        new_arcs = [" ".join(arc) for arc in new_arcs]
        new_arcs = "\n".join(new_arcs)

        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        return fst

    def remove_intra_word_blk(self, decoding_graphs, start_tokens, flag=True):
        if len(decoding_graphs.shape) == 2:
            decoding_graphs = k2.create_fsa_vec([decoding_graphs])
       
        num_fsas = decoding_graphs.shape[0]
        decoding_graph_list = []
        for i in range(num_fsas):
            decoding_graph_i = self._remove_intra_word_blk(decoding_graphs[i], start_tokens, flag=flag)
            decoding_graph_i = k2.connect(decoding_graph_i)
            decoding_graph_list.append(decoding_graph_i)
        
        decoding_graphs = k2.create_fsa_vec(decoding_graph_list)
        decoding_graphs = k2.arc_sort(decoding_graphs)
        decoding_graphs = decoding_graphs.to(self.device)
        return decoding_graphs

    # This works!
    def compile_ctc(
        self,
        piece_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        """Build a ctc graph from a list-of-list piece IDs.

        Args:
          piece_ids:
           IDs. It is a list-of-list integer 
          modified:
           See :func:`k2.ctc_graph` for its meaning.
        Return:
          Return an FsaVec, which is the result of composing a
          CTC topology with linear FSAs constructed from the given
          piece IDs.
        """
        graph = k2.ctc_graph(piece_ids, modified=modified, device=self.device)

        # graph = self.remove_intra_word_blk(graph, self.start_tokens, flag=self.remove_intra_word_blk_flag)
        return graph

    @staticmethod
    def hmm_topo(
        max_token: int,
        start_tokens: list,
        device = None,
        sil_id: int = 0,
    ) -> k2.Fsa:
        '''
        HMM topo
        '''
        print(f"Creating HMM topo for {max_token} tokens")
        num_tokens = max_token
        # assert (
        #     sil_id <= max_token
        # ), f"sil_id={sil_id} should be less or equal to max_token={max_token}"

        start_tokens = set(start_tokens)

        # ref: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py#L248

        start_state = 0
        loop_state = 1
        blk_state = 2
        next_available_state = 3
        arcs = []

        blk = sil_id
        arcs.append([start_state, start_state, blk, blk, 0])

        for i in range(1, max_token + 1):
            arcs.append([start_state, loop_state, i, i, 0])

        arcs.append([loop_state, blk_state, blk, blk, 0])
        arcs.append([blk_state, blk_state, blk, blk, 0])

        for i in range(1, max_token + 1):
            cur_state = next_available_state  # state_id
            next_available_state += 1

            arcs.append([loop_state, loop_state, i, i, 0])
            arcs.append([loop_state, cur_state, i, i, 0])
            arcs.append([cur_state, cur_state, i, blk, 0])
            arcs.append([cur_state, loop_state, i, blk, 0])
            
            arcs.append([start_state, cur_state, i, i, 0])

            if i in start_tokens:
                arcs.append([blk_state, loop_state, i, i, 0])
                arcs.append([blk_state, cur_state, i, i, 0])

        final_state = next_available_state
        next_available_state += 1
        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([loop_state, final_state, -1, -1, 0])
        arcs.append([blk_state, final_state, -1, -1, 0])    
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fst = k2.Fsa.from_str(arcs, acceptor=False)
        # fst = k2.remove_epsilon(fst)  # Credit: Matthew W
        # fst = k2.expand_ragged_attributes(fst)
        fst = k2.arc_sort(fst)
        
        if device is not None:
            fst = fst.to(device)

        return fst

    def compile_hmm(
        self,
        piece_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        transcript_fsa = k2.linear_fsa(piece_ids)
        transcript_fsa = k2.arc_sort(transcript_fsa)

        decoding_graphs = k2.compose(
            self.topo, transcript_fsa, treat_epsilons_specially=True
        )
        decoding_graphs = k2.connect(decoding_graphs)
        decoding_graphs = decoding_graphs.to(self.device)
        
        return decoding_graphs

    def compile(
        self,
        piece_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        if self.topo_type == "ctc":
            return self.compile_ctc(piece_ids, modified)
        elif self.topo_type == "hmm":
            return self.compile_hmm(piece_ids, modified)
        else:
            raise NotImplementedError

