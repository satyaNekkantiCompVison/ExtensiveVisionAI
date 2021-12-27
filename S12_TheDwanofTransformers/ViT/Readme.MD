The classes here correspond to the modules as described in the paper "Attention-is-all-you-need".

*Block*:
Block class implements the multi-head attention module. Instead of performing a single attention function with dmodel-dimensional keys, values and queries, the queries, keys and values are linearly projected h times with different, learned linear projections to dk, dk and dv dimensions, respectively.
Embeddings:

*MLP*:
MLP class implements a fully connected feed-forward-network. This is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.



*Attention* :
Attention function is described as mapping a query and a set of key-value pairs to an output. The query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. The function here implements "Scaled Dot-Product Attention" as shown in the figure 1. The input consists of queries and keys of dimension dk, and values of dimension dv. The dot products of the query with all keys, divide each by square root of dk, and apply a softmax function to obtain the weights on the values

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        

*Encoder*:
Encoder class implements the encoder module as described in the paper. Each encoder layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position wise fully connected feed-forward network. A residual connection is employed around each of the two sub-layers, followed by layer normalization.
