# LLVM_tutorials_llvm14

#### following https://releases.llvm.org/14.0.0/docs/tutorial/MyFirstLanguageFrontend/LangImpl03.html 


### From Standford Compiler Course on EdX:

"Well it consists five phases lexical analysis and parsing, which together take
care of the syntactic aspects of the language, semantic analysis, which, of
course, takes care of more semantic aspects, things like types and scope
rules. Optimization, Which is a collection of transformations on the program to
either make it run faster or use less memory. And finally code generation which
actually does the translation to another generation. And depending on our goals,
that translation might be to machine codes. It might be to a bite code for a
virtual machine. It might be to another high level programming language. Well
that's it for this lecture, and next time we'll pick up here and talk about these
five phases in more detail."


### JIT stands for "Just-In-Time" compilation, a technique used in the execution of computer programs. In a JIT compiler, the source code or bytecode is not compiled into machine code all at once. Instead, it is compiled into native machine code just before it is executed, often at runtime. This allows for certain optimizations that are not possible in a traditional ahead-of-time (AOT) compiled environment.
#### In the context of your job description and the tasks you'll be working on, here's what JIT compilation might involve:
1.	Dynamic Optimization: JIT allows for optimizations based on runtime behavior, which can be useful in adapting to different workloads dynamically.
2.	Code Specialization: Since JIT compilation happens at runtime, it can take advantage of the current execution context to generate specialized code that might be more efficient than general-purpose code.
3.	Reduced Startup Time: In some cases, JIT compilation can reduce the startup time for applications because only the code that is actually used needs to be compiled.
4.	Language Constructs: The job description mentioned "new language constructs in the front-end and middle-end optimizations." In a JIT compiler, these constructs can be optimized for better runtime performance.
5.	Integration with Frameworks: JIT compilation can be tightly integrated with machine learning frameworks like PyTorch and TensorFlow to dynamically compile computational graphs into optimized machine code. This is especially useful for optimizing custom operations in deep learning models, like the transformer models you might work with.
6.	GPU Acceleration: JIT can also be used to compile code for GPUs on-the-fly, allowing for more dynamic and adaptive utilization of GPU resources.
7.	Debugging and Profiling: JIT compilers often have features that enable easier debugging and profiling, allowing you to understand the performance characteristics of your code better.
Understanding JIT compilation and how to work with JIT compilers could be a valuable skill in your role, as it would give you the tools to optimize code for both CPU and GPU in a more flexible and adaptive manner.
 

### Potential Roles and Responsibilities:
1.	Optimization: Analyzing computational workloads and optimizing them by writing custom LLVM passes or CUDA kernels.
2.	Compiler Techniques: Implementing techniques to automatically parallelize and optimize the intermediate representation generated at runtime.
3.	Resource Virtualization: You may also be involved in adding a virtualization layer to decouple user applications from underlying hardware, making it easier to manage resources efficiently.
4.	Cross-Platform Support: The company is not restricted to NVIDIA GPUs but also supports AMD and NPUs. Your role may involve making sure the optimizations are effective across different types of hardware.
5.	AI Model Optimization: As the company also works on large-scale AI models, your role might involve optimizing these models for better performance and efficiency, especially when running on large clusters.
6.	Integration with ML Frameworks: Since the company's product is compatible with PyTorch and TensorFlow, you'll likely need to ensure that any compiler optimizations you develop are also compatible with these frameworks.
7.	Cloud Services: The company develops IaaS to AIaaS solutions, so you might also be involved in ensuring that the compiled and optimized code runs efficiently in a cloud-based environment.
### Interview Preparation:
1.	LLVM: Review the LLVM architecture, Intermediate Representation (IR), and how to write LLVM passes. Be prepared to discuss your experience with LLVM in your course.
2.	CUDA: Be prepared to discuss your experience with CUDA, the types of problems you've used it for, and any specific optimizations you've implemented.
3.	Parallel Computing: Brush up on your understanding of parallel algorithms, data structures, and how they can be implemented in CUDA and optimized with LLVM.
4.	Machine Learning: Gain a basic understanding of neural networks, especially the types of models like Large Language Models (LLMs) and transformer-based models (like BERT) that the company is likely working on. Know how these models are generally parallelized and optimized.
5.	Cloud Computing: Basic understanding of IaaS, PaaS, and SaaS as well as containerization technologies like Docker could be beneficial.
6.	System Design: Be prepared to discuss large-scale system design, as you would be working on optimizing code that runs on large clusters.
7.	Soft Skills: Don't underestimate the power of communication, problem-solving skills, and cultural fit. The job description emphasizes these as well.
Given your background in Computer Science and your CUDA certification, you already have a strong foundation. Your focus should be on bridging the gap between what you know and the specific technologies and approaches used by the company.


Day 1: Compiler Basics
•	Morning: Watch lectures on the basics of compilers from the Stanford EdX course. Focus on understanding the role of a compiler, and its different components like lexer, parser, and code generator.
•	Afternoon: Start with the LLVM "Kaleidoscope" tutorial. Make sure to set up LLVM on your local machine if you haven't done it already.
Day 2: LLVM Architecture and IR
•	Morning: Dive deeper into the LLVM architecture. Study what LLVM IR is and how it serves as a bridge between the front-end and back-end of a compiler.
•	Afternoon: Continue with the LLVM "Kaleidoscope" tutorial, focusing on generating LLVM IR.
Day 3: Writing LLVM Passes
•	Morning: Study how to write LLVM passes. Look for tutorials or documentation that explains this in detail.
•	Afternoon: Try to write a simple LLVM pass as practice. Review how optimizations are implemented as LLVM passes.
Day 4: Parallel Computing and Model Optimization
•	Morning: Research how machine learning models are parallelized. Focus on Large Language Models (LLMs) and transformer models like BERT.
•	Afternoon: Study optimization techniques for these models. Try to relate these with possible compiler optimizations.
Day 5: Virtualization and Resource Management
•	Morning: Study what resource virtualization means in the context of cloud computing and how it might be implemented in a compiler.
•	Afternoon: Explore how resource management works in large-scale clusters.
Day 6: PyTorch and TensorFlow Integration
•	Morning: Briefly go through how PyTorch and TensorFlow work, focusing on how models are compiled and run.
•	Afternoon: Study how LLVM can be integrated with these frameworks. Look for existing projects or papers that describe this.
Day 7: Mock Interview and Review
•	Morning: Have a mock interview with a friend or use a platform that allows for technical mock interviews. Focus on LLVM, CUDA, and system design questions.
•	Afternoon: Review everything you've learned. Go through your notes and make sure you're comfortable with all the major topics.
Optional: Throughout the Week
•	Soft Skills: Spare 30 minutes each day for soft skills. This could include practicing problem-solving or brushing up on your communication skills.


## Trace-based optimization 
is a technique often employed in Just-In-Time (JIT) compilers to optimize frequently executed paths, or "hot paths," in a program. The basic idea is to identify these hot paths at runtime, collect them into traces, and then optimize these traces as single units.
In the context of LLVM, you can build a trace-based optimizer by using LLVM's JIT compilation capabilities along with its rich set of program analysis and transformation passes. Here's a simplified view of how this might work:
1.	Trace Collection: As the program runs, the JIT compiler identifies hot paths in the code. These paths are sequences of instructions that are frequently executed. This could be done using performance counters, profiling, or other runtime metrics.
2.	Trace Compilation: Once a trace is identified, it's fed into LLVM's JIT, where it can be optimized as a single unit. This is where LLVM shines—you can leverage its existing passes for optimizations like constant folding, dead code elimination, loop unrolling, etc.
3.	Code Replacement: The optimized trace is then swapped in for the original code. Future executions will use the optimized trace, improving the program's performance.
4.	Monitoring and Adaptation: The runtime system continues to monitor code execution. If the behavior changes (i.e., a different path becomes hot), the JIT compiler can invalidate the old optimized trace and generate a new one.
Some specific kinds of trace-based optimizations you might implement include:
•	Instruction Reordering: Reorder instructions to minimize pipeline stalls.
•	Register Allocation: Optimize the usage of registers within the trace.
•	Loop Unrolling: For loops that are entirely within a hot path, unroll them to decrease the overhead of the loop control code.
•	Inlining: Function calls that are frequently encountered in hot paths can be inlined to eliminate the call overhead.
•	Dead Code Elimination: Remove computations whose results are not used later in the hot path.
LLVM's modular architecture and extensive set of optimization passes make it a powerful tool for building sophisticated trace-based optimizers.
トレースベース最適化は、Just-In-Time（JIT）コンパイラで頻繁に実行されるパス、つまりプログラム内の「ホットパス」を最適化するためによく使用される技術です。 基本的な考え方は、実行時にこれらのホット パスを特定し、トレースに収集してから、これらのトレースを単一ユニットとして最適化することです。
LLVM のコンテキストでは、LLVM の JIT コンパイル機能と豊富なプログラム分析および変換パスを使用して、トレースベースのオプティマイザを構築できます。 以下は、これがどのように機能するかについての簡単な見解です:
1.	トレース収集: プログラムが実行されると、JIT コンパイラはコード内のホット パスを識別します。 これらのパスは、頻繁に実行される命令のシーケンスです。 これは、パフォーマンス カウンタ、プロファイリング、またはその他のランタイム メトリックを使用して実行できます。
2. トレースコンパイル: トレースが識別されると、LLVM の JIT に入力され、単一のユニットとして最適化できます。 ここで LLVM は輝きます。既存のパスを利用して、継続的なフォールディング、デッドコード除去、ループアンローリングなどの最適化を行うことができます。
3.	コード置換: 次に、最適化されたトレースが元のコードとスワップインされます。 今後の実行では、最適化されたトレースが使用され、プログラムのパフォーマンスが向上します。
4. 監視と適応: 実行時システムは、コードの実行を監視し続けます。 動作が変化すると（つまり、別のパスがホットになる場合）、JIT コンパイラは古い最適化トレースを無効にして新しいトレースを生成できます。
実装できる特定の種類のトレースベース最適化には、次のようなものがあります:
• 命令の並べ替え: パイプラインの停止を最小限に抑えるように命令を並べ替えます。
• レジスタ割り当て: トレース内のレジスタの使用を最適化します。
• ループ解除: 完全にホット パス内にあるループの場合は、ロールを解除してループ制御コードのオーバーヘッドを減らします。
• インライン: ホット パスで頻繁に発生するファンクション コールをインライン化して、コール オーバーヘッドを解消できます。
• デッドコードの削除: 後でホットパスで結果が使用されない計算を削除します。
LLVM のモジュラー型アーキテクチャと広範な最適化パスは、高度なトレースベースのオプティマイザを構築するための強力なツールです。
트레이스 기반 최적화는 Just-In-Time(JIT) 컴파일러에서 자주 실행되는 경로, 즉 프로그램 내 '핫패스'를 최적화하기 위해 자주 사용되는 기술입니다. 기본적인 생각은 실행 시 이러한 핫입니다. 경로를 특정하고 트레이스에 수집한 후 이러한 트레이스를 단일 유닛으로 최적화하는 것입니다.
LLVM 컨텍스트에서는 LLVM의 JIT 컴파일 기능과 풍부한 프로그램 분석 및 변환 경로를 사용하여 트레이스 기반 옵티마이저를 구축할 수 있습니다. 다음은 이것이 어떻게 작동하는지에 대한 간단한 견해입니다:
1. 추적 수집입니다: 프로그램이 실행되면 JIT 컴파일러는 코드 내 핫합니다. 경로를 식별합니다. 이러한 경로는 자주 실행되는 명령의 시퀀스입니다. 이것은 퍼포먼스입니다. 카운터, 프로파일링 또는 기타 런타임입니다. 메트릭을 사용하여 실행할 수 있습니다.
2. 트레이스 컴파일입니다. 트레이스가 식별되면 LLVM의 JIT에 입력되어 단일 유닛으로 최적화할 수 있습니다. 여기서 LLVM은 빛납니다.기존 경로를 이용하여 지속적인 폴딩, 데드코드 제거, 루프 언롤링 등의 최적화를 수행할 수 있습니다.
3. 코드 치환: 그런 다음 최적화된 트레이스가 원래 코드와 스왑 인됩니다. 향후 실행에서는 최적화된 트레이스가 사용되어 프로그램의 성능이 향상됩니다.
4. 감시와 적응: 실행 시 시스템은 코드의 실행을 계속 감시합니다. 동작이 변화하면(즉, 다른 경로가 핫해지는 경우) JIT 컴파일러는 오래된 최적화 트레이스를 비활성화하여 새로운 트레이스를 생성할 수 있습니다.
구현할 수 있는 특정 종류의 트레이스 기반 최적화에는 다음과 같은 것이 있습니다:
• 명령 정렬: 파이프라인 정지를 최소화하도록 명령을 정렬합니다.
• 레지스터 할당: 트레이스 내 레지스터 사용을 최적화합니다.
• 루프 해제: 완전히 핫합니다 경로 내에 있는 루프의 경우 롤을 해제하여 루프 제어 코드의 오버헤드를 줄입니다.
• 인라인입니다. 핫패스에서 자주 발생하는 펑션입니다. 콜을 인라인화하여 콜 오버헤드를 해소할 수 있습니다.
• 데드코드 삭제: 나중에 핫패스에서 결과가 사용되지 않는 계산을 삭제합니다.
LLVM의 모듈형 아키텍처와 광범위한 최적화 경로는 고급 트레이스 기반 옵티마이저를 구축하기 위한 강력한 도구입니다.


#### Does that involve instrumenting code to identify hot paths?
#### ChatGPT
Yes, identifying hot paths usually involves some form of code instrumentation or profiling. The goal is to gather data on which parts of the code are executed most frequently. Here are a few common methods for identifying hot paths:
1.	Dynamic Instrumentation: Inserting extra code at runtime to track the frequency of function calls, loop iterations, or even individual instructions. Tools like Intel's Pin or DynamoRIO can be used for this kind of dynamic binary instrumentation.
2.	Static Instrumentation: Modifying the source code to include counters or timers before it is compiled. This is less flexible than dynamic instrumentation but may incur lower overhead.
3.	Sampling Profilers: These tools interrupt the program at regular intervals and record the program counter. By analyzing where the program spends most of its time, you can identify hot paths. Examples include perf on Linux and Instruments on macOS.
4.	Hardware Counters: Some CPUs offer performance counters that can track various metrics like cache misses, branch mispredictions, etc. Tools like perf can access these counters to help identify hot paths.
5.	Custom Profiling: In some cases, especially for domain-specific applications or algorithms, custom profiling logic is added to the code to track how often certain functions or code paths are executed.
6.	Feedback from Previous Runs: For JIT compilers, feedback from previous executions can be used to inform which paths are likely to be hot in the future.
After identifying the hot paths, the next step in trace-based optimization would be to feed these traces into your optimization pipeline, which could be built using LLVM's various passes and utilities.
 
#### What if it's Cuda code? Can I use info from reports output with `nsys` command to NSight for visual inspection?
#### ChatGPT
Yes, NVIDIA's Nsight tools and the nsys profiling utility provide invaluable information for optimizing CUDA code. They can show you detailed timing and execution behavior of your CUDA kernels, memory transfers, and API calls, among other things. Specifically, you can look at:
1.	Kernel Execution Time: This is the most straightforward metric and tells you which kernels are taking the most time.
2.	Memory Transactions: Knowing the number of global and shared memory transactions can help you optimize memory access patterns.
3.	Warp Occupancy: This metric can tell you if you're making the most of the GPU's computational resources.
4.	Instruction Statistics: You can see which instructions are most frequently executed in your code.
5.	Stream and Concurrency Information: Nsight will show you how well you are overlapping computation with data transfers, which is key for performance on a GPU.
6.	API Calls: Sometimes the bottleneck can be CPU-side API calls rather than the GPU kernels themselves.
7.	Divergent Branches: Conditional statements that cause threads within the same warp to follow different code paths can severely impact performance.
The Nsight Visual Profiler and the Compute Sanitizer can also help identify issues like data races, invalid accesses, etc., which might not directly relate to performance but are crucial for correctness.
By combining this data with your knowledge of the application's computational graph and algorithms, you can identify the "hot paths" in your CUDA code. You can then focus your optimization efforts on these critical sections, potentially by rewriting them in a more efficient manner, reordering operations to improve memory access patterns, or even offloading certain parts to different parts of your heterogeneous system.
So, if you're working on optimizing CUDA code, understanding how to interpret and act upon the data from Nsight tools can be crucial.

