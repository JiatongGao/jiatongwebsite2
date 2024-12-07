function loadContent(section) {
    const content = document.getElementById("content");
    const hero = document.getElementById("hero"); // 获取照片区域

    if (section === "about") {
        hero.style.display = "block"; // 显示照片区域
        content.innerHTML = `
            <div class="about-section">
                <div class="about-content">
                    <a href="Jiatong Gao-6.7 cl.pdf" target="_blank" class="resume-link">
                        <img src="icebear.gif" alt="Profile Image" class="profile-image">
                    </a>
                    <p class="cv-text">Touch this bear for my CV</p>
                </div>
                <div class="about-text">
                    <h1 class="name-title">Jiatong Gao</h1>
                    <p>Hey! I'm Jiatong Gao, and I am currently a senior in the Department of Mathematics at the University of Washington.</p>
                    <p> I aim to be a data scientist who passionate about transforming complex data into meaningful insights that drive decision-making and innovation. 
                    With a strong foundation in mathematics and coding skills, I approach data challenges with analytical rigor and clarity. 
                    My work often intersects machine learning and strategic thinking, aiming to uncover trends and patterns that empower businesses to act confidently in an ever-evolving market. 
                    I value accuracy, ethical use of data, and the clarity needed to communicate findings to diverse audiences. 
                    This portfolio reflects my journey, from foundational projects to advanced applications, embodying my dedication to responsible and impactful data science. </p>
                    </p>
                    <p>In my free time, I'm a lover of life, I enjoy photography, music, and all things sports.</p>
                </div>
            </div>
            <div class="social-icons">
            <a href="https://www.linkedin.com/in/jiatong-gao-56624622a/" target="_blank" class="social-icon"><i class="fab fa-linkedin"></i></a>
            <a href="https://www.instagram.com/jiatong58/" target="_blank" class="social-icon"><i class="fab fa-instagram"></i></a>
            </div>
            <div class="education-section">
                <img src="Washington_Huskies_logo.svg.png" alt="University of Washington Logo" class="education-logo">
                <div class="education-content">
                    <h2>Education</h2>
                    <div class="degree">
                        <a href="https://math.washington.edu/" target="_blank" class="degree-link">BS of Mathematics</a>
                        <p><a href="https://www.washington.edu" target="_blank" class="school-link">University of Washington</a></p>
                    </div>
                </div>
            </div>
            <div class="research-section">
            <h2>Current Research</h2>
                <div class="research-card">
                    <h3>Experimental Lean Lab-WXML</h3>
                    <p><strong>Advisor:</strong> Prof. Jarod Alper</p>
                    <p>We use Lean 4 to write strict mathematical proof for some exercises and theorems. More info for Lean, follow this link: <a href="https://leanprover-community.github.io/learn.html" target="_blank">Lean 4</a> </p>
                    
                    <p>During my current research, I integrated Lean 4, a formal proof assistant, to validate and formalize several core examples and counterexamples from commutative algebra. 
                    My primary goal was to leverage Lean 4 to formalize definitions and prove properties associated with modules, ideals, and ring homomorphisms—concepts foundational to understanding commutative structures. </p>

                    <p>One of the primary advantages of Lean 4 was its enhanced metaprogramming capabilities, which allowed me to construct proofs in a more modular and readable manner. For example, I used Lean to formalize the definitions of prime ideals and maximal ideals, proceeding then to verify the well-known correspondence between these ideals and certain quotient rings. 
                    By structuring proofs in Lean, I was able to test the consistency of assumptions and gain a deeper understanding of how various definitions interact in a commutative setting.</p>
                    <div class="research-tags">
                        <span class="tag">Mathematical Proof</span>
                        <span class="tag">Lean 4</span>
                        <span class="tag">Commutative Algebra</span>
                    </div>
                </div>
            </div>
            <div class="skills-section">
                <h2>Skills</h2>
                <div class="skills-category">
                    <h3>Tools</h3>
                    <div class="skills-container tools-skills">
                        <div class="skill" onclick="openModal('python')">Python</div>
                        <div class="skill" onclick="openModal('Rstudio')">R Studio</div>
                        <div class="skill" onclick="openModal('SQl')">SQL</div>
                    </div>
                </div>
                <div class="skills-category">
                    <h3>Application Skills</h3>
                    <div class="skills-container application-skills">
                        <div class="skill" onclick="openModal('machineLearning')">Machine Learning</div>
                        <div class="skill" onclick="openModal('quantitativeFinance')">Quantitative Finance</div>
                    </div>
                </div>
                <div class="skills-category">
                    <h3>Math</h3>
                    <div class="skills-container math-skills">
                        <div class="skill" onclick="openModal('linearalgebra')">Advanced Linear Algebra</div>
                        
                        <div class="skill" onclick="openModal('realcomplexAnalysis')">Real/Complex Analysis</div>
                        <div class="skill" onclick="openModal('DifferentialGeo')">Differential Geometry</div>
                        <div class="skill" onclick="openModal('Probability')">Probability</div>
                    </div>
                </div>
            </div>

            <!-- Modals for each skill -->
            <div id="python-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('python')">&times;</span>
                    <h3>Python</h3>
                    <div class="skill-section">
                        <h4>NumPy:</h4>
                        <p>Proficient in NumPy, widely applied for efficient array and matrix computations. 
                        Used NumPy for data manipulation and mathematical operations in numerous data science and machine learning projects, 
                        providing a powerful mathematical framework to support complex scientific computations.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Pandas:</h4>
                        <p>Expert in Pandas, the go-to library for handling and analyzing structured data. Applied Pandas in financial data analysis, time series forecasting,
                        and cross-domain data integration projects, leveraging its powerful data processing capabilities including data cleaning, transformation, and aggregation.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Matplotlib/Seaborn:</h4>
                        <p>Skilled in using Matplotlib for data visualization. Designed and generated various customized charts using Matplotlib, effectively supporting the presentation and interpretation of data analysis results,
                        helping teams and clients intuitively understand data trends and insights. Also proficient in drawing the interactive plot by Seaborn</p>
                    </div>
                </div>
            </div>
            
            <div id="machineLearning-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('machineLearning')">&times;</span>
                    <h3>Machine Learning</h3>
                    <div class="skill-section">
                        <h4>Tensorflow:</h4>
                        <p>Experienced with tensorflow for constructing models. 
                        Utilized TensorFlow for model training and optimization in research for solving solution of partial differential equations.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Pytorch:</h4>
                        <p>Familiar with the functions in Pytorch. Proficient in implementing neural networks, including convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers.
                        Experienced in leveraging GPU acceleration for faster training of large models.
                        Hands-on experience with PyTorch’s optimizers and schedulers for fine-tuning models, such as SGD, Adam, and learning rate annealing.</p>
                    </div>
                    <div class="skill-section">
                        <h4>XGBoots:</h4>
                        <p>Implemented XGBoost in the automated trading code section of the trading competition to analyze existing information and predict next-day market conditions.
                        Effectively applied in complex financial data analysis and market trend prediction.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Scikit-learn:</h4>
                        <p>Extensive use of Scikit-learn’s preprocessing utilities (e.g., scaling, encoding, feature selection) to prepare datasets.
                        Built models using algorithms like linear regression, logistic regression, decision trees, and random forests.Experienced in clustering like K-Means and dimensionality reduction techniques.
                        Applied cross-validation, grid search, F1-score, and MAE for robust model evaluation.
                        Seamlessly combined Scikit-learn with Pandas and NumPy for end-to-end data analysis and machine learning pipelines.</p>
                    </div>
                </div>
            </div>

            <div id="machineLearning-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('machineLearning')">&times;</span>
                    <h3>Machine Learning</h3>
                    <div class="skill-section">
                        <h4>Tensorflow:</h4>
                        <p>Experienced with tensorflow for constructing models. 
                        Utilized TensorFlow for model training and optimization in research for solving solution of partial differential equations.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Pytorch:</h4>
                        <p>Familiar with the functions in Pytorch. Proficient in implementing neural networks, including convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers.
                        Experienced in leveraging GPU acceleration for faster training of large models.
                        Hands-on experience with PyTorch’s optimizers and schedulers for fine-tuning models, such as SGD, Adam, and learning rate annealing.</p>
                    </div>
                    <div class="skill-section">
                        <h4>XGBoots:</h4>
                        <p>Implemented XGBoost in the automated trading code section of the trading competition to analyze existing information and predict next-day market conditions.
                        Effectively applied in complex financial data analysis and market trend prediction.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Scikit-learn:</h4>
                        <p>Extensive use of Scikit-learn’s preprocessing utilities (e.g., scaling, encoding, feature selection) to prepare datasets.
                        Built models using algorithms like linear regression, logistic regression, decision trees, and random forests.Experienced in clustering like K-Means and dimensionality reduction techniques.
                        Applied cross-validation, grid search, F1-score, and MAE for robust model evaluation.
                        Seamlessly combined Scikit-learn with Pandas and NumPy for end-to-end data analysis and machine learning pipelines.</p>
                    </div>
                </div>
            </div>

            <div id="SQl-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('SQl')">&times;</span>
                    <h3>SQL</h3>
                    <div class="skill-section">
                        <p>Skilled in writing complex SQL queries for extracting and analyzing data from large datasets。 Proficient in using aggregate functions, subqueries, and common table expressions (CTEs) to simplify and optimize queries.Created and managed relational databases for projects using MySQL connecting with Microsoft Azure.
                        Integrated SQL queries with Python for advanced data analysis workflows</p>
                    </div>
                </div>
            </div>
            <div id="linearalgebra-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('linearalgebra')">&times;</span>
                    <h3>Advanced Linear Algebra</h3>
                    <div class="skill-section">
                        <h4>Matrix Decomposition</h4>
                        <p> Proficient in Singular Value Decomposition (SVD), LU decomposition, QR decomposition, and Eigen decomposition. 
                        Skilled in applying these techniques to solve systems of linear equations, perform dimensionality reduction, and analyze data.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Eigenvalues and Eigenvectors</h4>
                        <p> Deep understanding of eigenvalues, eigenvectors, 
                        and their applications in stability analysis, spectral clustering, and machine learning algorithms like Principal Component Analysis (PCA)</p>
                    </div>
                    <div class="skill-section">
                        <h4>Positive Definite Matrices</h4>
                        <p>Applied properties of symmetric, positive-definite matrices in optimization and machine learning (e.g., covariance matrices in multivariate analysis).</p>
                    </div>
                </div>
            </div>
            <div id="quantitativeFinance-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('quantitativeFinance')">&times;</span>
                    <h3>Quantitative Finance</h3>
                    <div class="skill-section">
                        <h4>Financial Modeling</h4>
                        <p>Developed mathematical models for financial risk management and pricing derivatives during academic projects.
                        Familiar with the Black-Scholes model and Monte Carlo simulations to compute option pricing and analyze market scenarios.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Risk Management</h4>
                        <p>Modeled risk exposure through Value at Risk (VaR) and stress testing, simulating potential losses under extreme market conditions. 
                        Designed strategies to hedge against risk in portfolios by using derivatives like options and futures.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Algorithmic Trading</h4>
                        <p>Developed and implemented automated trading strategies using Python. 
                        Designed and backtested trading algorithms based on statistical arbitrage, momentum, and mean reversion strategies.
                        Integrated real-time market data feeds to execute trades programmatically with minimal latency.
                    </div>
                </div>
            </div>
            <div id="realcomplexAnalysis-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('realcomplexAnalysis')">&times;</span>
                    <h3>Real and Complex Analysis</h3>
                    <div class="skill-section">
                        <h4>Real Analysis<h4>
                            <h5>Lebesgue Integration</h5>
                            <p> Comprehensive understanding of Lebesgue integration, focusing on its advantages over Riemann integration, including better handling of discontinuous functions and convergence. 
                            Expertise in applying the Dominated Convergence Theorem and Monotone Convergence Theorem for evaluating limits of integrals.</p>
                            <h5>Measure Theory</h5>
                            <p>Proficient in foundational concepts such as σ-algebras, measurable functions, and the construction of measures (e.g., Lebesgue measure). 
                            Skilled in proving and applying key theorems like Fubini’s Theorem to analyze multi-dimensional integration and transformations.</p>
                            <h5>Functional Analysis Foundations</h5>
                            <p>Familiarity with the spaces Lp, including their norms, convergence properties, and dual spaces. 
                            Applications of Banach and Hilbert spaces in solving optimization and variational problems.</p>
                        <h4>Complex Analysis</h5>
                            <h5>Contour Integration</h5>
                            <p>Mastery of contour integration and its theoretical foundations, such as Cauchy’s Integral Formula and its extensions.</p>
                            <h5>Series Representations</h5>
                            <p>Thorough knowledge of Taylor and Laurent series for representing complex functions and understanding their convergence properties. 
                            Applications of these series in identifying singularities and their classifications</p>
                            <h5>Conformal Mappings</h5>
                            <p>Advanced understanding of conformal mappings and their properties, including the Riemann Mapping Theorem.
                            Applications to solve boundary value problems in mathematical physics.</p>
                    </div>
                </div>
            </div>
            <div id="DifferentialGeo-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('DifferentialGeo')">&times;</span>
                    <h3>Differential Geometry</h3>
                    <div class="skill-section">
                        <h4> Curves in space</h4>
                        <p>Studied parametrized curves, their tangent vectors, arc length, and smoothness conditions.<br>
                        Mastered Frenet formulas to analyze the curvature and torsion of space curves, with applications in 3D geometry and physics. <br>
                        Explored global properties of plane curves, such as the isoperimetric inequality and the four-vertex theorem. <br>
                        Gained proficiency in the Local Canonical Form of curves, understanding how curvature uniquely determines the shape of curves up to rigid motion.</p>
                    </div>
                    <div class="skill-section">
                        <h4> Regular Surface</h4>
                        <p>Acquired a deep understanding of regular surfaces, focusing on their differentiable structures and parametrizations. <br>
                        Proficient in computing tangent planes, differentials, and the First Fundamental Form for intrinsic measurements like distance, angle, and area.<br>
                        Studied oriented surfaces and their applications to surface integrals and flux computation.<br>
                        Explored the concept of compact orientable surfaces and their geometric properties.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Gauss Map</h4>
                        <p>Mastered the Gauss map and its role in describing the curvature of surfaces. <br>
                        Studied Gaussian curvature, mean curvature, and their classifications of surfaces.<br>
                        Analyzed minimal surfaces and ruled surfaces using the Gauss map.<br>
                        Investigated relationships between the Gauss map, principal curvatures, and the Second Fundamental Form.
                    </div>
                    <div class="skill-section">
                        <h4>Intrinsic Geometry of Surfaces</h4>
                        <p>Gained expertise in intrinsic properties such as geodesics, isometries, and conformal maps.<br>
                        Studied parallel transport, holonomy, and their roles in understanding curvature.<br>
                        Mastered the Gauss-Bonnet theorem and its applications to the topology of compact orientable surfaces and Euler characteristics.<br>
                        Applied the exponential map to study geodesic polar coordinates and convex neighborhoods.
                    </div>
                    <div class="skill-section">
                        <h4>Global Differential Geometry</h4>
                        <p>Investigated global theorems like the rigidity of the sphere and Hopf-Rinow theorem. <br>
                        Analyzed Jacobi fields, conjugate points, and variational principles for understanding geodesics.
                    </div>
                </div>
            </div>

            <div id="Rstudio-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('Rstudio')">&times;</span>
                    <h3>R studio</h3>
                    <div class="skill-section">
                        <h4>Data Analysis and Manipulation</h4>
                        <p>Proficient in using R Studio for comprehensive data analysis, including data cleaning, transformation, and aggregation..<br>
                        Leveraged packages like dplyr, tidyr, and data.table to preprocess large datasets efficiently. <br>
                        Applied advanced data manipulation techniques to reshape data, handle missing values, and merge multiple data sources.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Statistical Modeling</h4>
                        <p>Developed a strong command of statistical modeling techniques using R, including linear regression, generalized linear models (GLMs), and time series forecasting.<br>
                        Applied lm(), glm(), and forecast packages to analyze and predict trends in financial and scientific datasets.<br>
                        Conducted hypothesis testing, ANOVA, and goodness-of-fit analyses to validate models.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Data Visualization</h4>
                        <p>Skilled in creating compelling and interactive visualizations with ggplot2, plotly, and shiny.<br>
                        Designed custom charts such as scatter plots, histograms, bar charts, and heatmaps to communicate insights effectively.<br>
                        Proficient in visualizing high-dimensional data using techniques like PCA plots, correlation matrices, and boxplots.
                    </div>
                    <div class="skill-section">
                        <h4>Machine Learning and Statistical Computing</h4>
                        <p>Experienced in implementing machine learning models in R, including random forests, decision trees, and clustering algorithms.<br>
                        Conducted dimensionality reduction techniques such as PCA and t-SNE to explore complex datasets.<br>
                        Optimized predictive models with cross-validation and hyperparameter tuning.</p>
                    </div>
                </div>
            </div>
            
            <div id="Probability-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('Probability')">&times;</span>
                    <h3>Probability</h3>
                    <div class="skill-section">
                        <h4>Measure-Theoretic Foundations</h4>
                        <p>Deep understanding of measure-theoretic probability, constructing probability spaces using σ-algebras and measures.<br>
                        Proficient in formalizing random variables as measurable functions and analyzing their distributions using pushforward measures. <br>
                        Expertise in working with Lebesgue integrals to compute expectations and prove properties of random variables.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Stochastic Processes</h4>
                        <p>Studied stationary distributions, spectral decomposition of transition matrices, and ergodicity in Markov chains.<br>
                        Modeled complex random systems using Poisson processes, focusing on applications in queuing theory and telecommunications.<br>
                        Applied advanced stochastic calculus, including Itô’s Lemma, to analyze Brownian motion and diffusion processes.</p>
                    </div>
                    <div class="skill-section">
                        <h4>Conditional Expectations and Martingales</h4>
                        <p>Proficient in the theory of conditional expectations, proving key properties and applications in optimal prediction.<br>
                        Extensive study of martingales, including Doob’s martingale inequalities and stopping time theorems.<br>
                        Applied martingales to analyze financial models, betting strategies, and stochastic optimization. </p>
                    </div>
                </div>
            </div>





        `;
    } else {
        hero.style.display = "none"; // 隐藏照片区域
        if (section === "experience") {
            content.innerHTML = `
                <div class="experience-page">
                    <div class="experience-card" onclick="openExperienceModal('project1')">
                        <img src="lean_logo2.png" alt="Project 1 Image" class="experience-image">
                        <div class="overlay1">
                            <h2>Experimental Lean Lab</h2>
                            <p>Use Lean 4 to write strict mathematical proof for some exercises and theorems</p>
                        </div>
                    </div>
                    <div class="experience-card" onclick="openExperienceModal('project2')">
                        <img src="fluids-04-00121-g012.webp" alt="Project 2 Image" class="experience-image">
                        <div class="overlay2">
                            <h2>Numerical Solution of Nonlinear Schrödinger Equation</h2>
                            <p>Find the numerical solution of NLS on 3-dimensional space by machine learning</p>
                        </div>
                    </div>
                    <div class="experience-card" onclick="openExperienceModal('project3')">
                    <img src="860_main_weather_and_prediction.png" alt="Project 3 Image" class="experience-image">
                        <div class="overlay3">
                            <h2>Predictive Analytics for Weather Type</h2>
                            <p>Make a prediction of future weather type based on perfect dataset</p>
                        </div>
                    </div>
                    <div class="experience-card" onclick="openExperienceModal('project4')">
                        <img src="0_haiF8hz3FNGYMxHV.jpg" alt="Project 4 Image" class="experience-image">
                        <div class="overlay4">
                            <h2>Rotman International Trading Competition</h2>
                            <p>Participate an worldwide Algo Trading Competition</p>
                        </div>
                    </div>
                    <div class="experience-card" onclick="openExperienceModal('project5')">
                        <img src="images.png" alt="Project52 Image" class="experience-image">
                        <div class="overlay5">
                            <h2>Comparative Analysis of Asset Pricing Models</h2>
                            <p>Analysis of different factors models</p>
                        </div>
                    </div>
                </div>      
                </div>          
                <div id="experience-modal" class="experience-modal">
                    <div class="experience-modal-content">
                        <span class="experience-modal-close" onclick="closeExperienceModal()">&times;</span>
                        <div id="experience-modal-body"></div>
                    </div>
                </div>                


                       


            `;
        } 
    }
}
function openModal(skillId) {
    document.getElementById(skillId + '-modal').style.display = 'block';
}

function closeModal(skillId) {
    document.getElementById(skillId + '-modal').style.display = 'none';
}
function openExperienceModal(projectId) {
    // Get the modal and modal body
    const modal = document.getElementById('experience-modal');
    const modalBody = document.getElementById('experience-modal-body');

    // Define project content dynamically
    const projects = {
        project1: {
            title: "Experimental Lean Lab-WXML(On going)",
            description: `
                <p>This project explores I integrated Lean 4, a formal proof assistant, to validate and formalize several core examples and counterexamples from commutative algebra. 
                My primary goal was to leverage Lean 4 to formalize definitions and prove properties associated with modules, ideals, and ring homomorphisms—concepts foundational to understanding commutative structures.</p>
                <p>One of the primary advantages of Lean 4 was its enhanced metaprogramming capabilities, which allowed me to construct proofs in a more modular and readable manner. For example, I used Lean to formalize the definitions of prime ideals and maximal ideals, proceeding then to verify the well-known correspondence between these ideals and certain quotient rings.
                By structuring proofs in Lean, I was able to test the consistency of assumptions and gain a deeper understanding of how various definitions interact in a commutative setting.</p>
            `,
        },
        project2: {
            title: "Numerical Solution of Nonlinear Schrödinger Equation",
            description: `
                <p>This project involves utilizing advanced machine learning techniques to explore solutions to a highly complex mathematical equation. 
                    Under the guidance of Professor Xueying Yu, this research focuses on solving the norm of solutions to the Nonlinear Schrödinger Equation (NLS) within a three-dimensional domain. 
                    By leveraging Deep Neural Networks (DNNs) and Physics-Informed Neural Networks (PINNs), the project aims to capture the intricate nonlinear relationships inherent in these equations, contributing to a deeper understanding of mathematical and physical phenomena.
                    The work entails establishing and training a DNN model using Python. 
                    This involves defining the model architecture, setting up a robust training pipeline, and designing loss functions tailored to the problem at hand. 
                    The research includes experimenting with various activation functions to assess their impact on the model's performance, as well as applying rigorous hyperparameter tuning to optimize results. 
                    These steps are critical to ensure the model accurately represents the behavior of the NLS in the targeted domain.
                    An intriguing aspect of the project is the observation of the model's periodic results in lower-dimensional cases. 
                    This phenomenon provided a basis for further analysis to ensure consistency when extending the methodology to higher dimensions. 
                    Such theoretical validation is crucial to maintaining the integrity of the results and expanding the applicability of the research to more complex scenarios. 
                    And for the higher and more complex situation, we need to use supercomputer, because it involves with nearly billons of points
                </p>
                <p><strong>Highlights:</strong></p>
                <ul>
                    <li>Developed a PINN model to solve the Nonlinear Schrödinger Equation in a three-dimensional space. </li>
                    <li>Find the periodical solutions on lower dimension.</li>
                    <li>Need supercomputer work for higher dimensions.</li>
                </ul>
                <a href="https://github.com/JiatongGao/NLSsolution" target="_blank" id="nls-link">Here is the Github Link</a>
            `,
        },
        project3: {
            title: "Predictive Analytics for Weather Type",
            description: `
                <p>The Predictive Analytics for Weather Type project focuses on leveraging machine learning techniques to forecast weather types under various meteorological conditions.
                As an independent project, it involved analyzing extensive meteorological datasets and exploring the relationships between factors such as temperature, UV index, and other environmental variables. 
                Data preprocessing included standardization using tools like StandardScaler and visualizing correlations between features using Seaborn, ensuring a robust understanding of the data structure.
                The project implemented multiple machine learning models, including Logistic Regression, Decision Trees, Random Forests, and K-Nearest Neighbors (KNN), utilizing Scikit-Learn for model training and evaluation. 
                To further enhance predictive capabilities, a neural network was built with PyTorch, incorporating fully connected layers, batch normalization, and ReLU activation functions. 
                This step significantly improved the model's ability to generalize across unseen data.
                Optimization efforts included adjusting the neural network's architecture and hyperparameters, iterating over 1,500 training epochs to achieve the best possible performance. 
                As a result, the model successfully predicted weather types with an impressive accuracy of 91%. 
                This outcome demonstrates the project's effectiveness in combining traditional machine learning methods with deep learning techniques to tackle complex predictive tasks in meteorology.

                </p>
                <p><strong>Highlights:</strong></p>
                <ul>
                    <li>Utilized machine learning techniques to accurately predict weather types based on meteorological data. </li>
                    <li>Improved neural network generalization using PyTorch to enhance performance..</li>
                    <li>Achieved a 91% accuracy in weather type predictions. </li>
                </ul>
                <a href="project.pdf" target="_blank" class="pdf-link">Project Report</a>
            `,
        },
        project4: {
            title: "Rotman International Trading Competition",
            description: `
                <p>The Rotman International Trading Competition involved developing and optimizing algorithmic trading strategies for various financial cases, including Exchange-Traded Funds (ETF) and Commodity Futures. 
                Leveraging Python, the team implemented machine learning models such as Linear Regression and Random Forest to analyze trading patterns and generate effective buy/sell signals to maximize returns. 
                By extracting and generating technical indicators, the project enhanced the models' predictive capabilities, enabling effective monitoring of market price volatility and trading volume. 
                Furthermore, the integration of multiple trading algorithms, including trend-following and mean-reversion strategies, allowed the team to adapt to volatile market conditions and gain competitive advantages.
                The competition emphasized the importance of balancing theoretical knowledge with practical application, as participants applied data-driven approaches to real-world trading scenarios. 
                Through extensive backtesting and strategy refinement, the project demonstrated how innovative algorithmic solutions could improve investment performance. This experience also highlighted the value of teamwork, problem-solving,
                and staying adaptive in dynamic financial markets.
                </p>
                <p><strong>Highlights:</strong></p>
                <ul>
                    <li>Design and optimize algorithmic trading strategies </li>
                    <li>Enhanced predictive accuracy by utilizing machine learning models</li>
                    <li>Integrated advanced algorithms to effectively navigate and capitalize on volatile market conditions. </li>
                </ul>
                <a href="https://medium.com/@jlin0109/rotman-international-trading-competition-2024-experience-cfc30d180d1b" target="_blank" id="nls-link">Competition Summary of our team(Written by my teammate)</a>
            `,
        },
        project5: {
            title: "Comparative Analysis of Asset Pricing Models",
            description: `
                <p>The project on Comparative Analysis of Asset Pricing Models investigated the effectiveness of various models in predicting stock prices, specifically focusing on the Capital Asset Pricing Model (CAPM), the Fama-French Three-Factor Model, and the Fama-French Five-Factor Model. 
                The analysis incorporated diverse portfolio strategies that accounted for factors like company size, book-to-market ratio, profitability, and investment patterns, enabling a robust evaluation under varying market conditions. 
                By conducting in-depth comparisons, the study identified the Five-Factor Model as the superior predictor, particularly in challenging scenarios involving micro-cap stocks and low-profitability firms.
                This project offered valuable insights into the practical applications and limitations of popular asset pricing models, providing a deeper understanding of how multifactor approaches can enhance prediction accuracy.
                The results highlighted the importance of incorporating additional economic factors into financial modeling to improve investment strategies and risk assessment in complex market environments.
                </p>
                <p><strong>Highlights:</strong></p>
                <ul>
                    <li>Conducted an in-depth comparison between models and evaluate their predictive performance for stock prices. </li>
                    <li>Implemented diverse portfolio strategies to analyze model performance across market conditions.</li>
                </ul>
                <a href="Cfrm_425.pdf" target="_blank" class="pdf-link">Project Report</a>
            `,
        },
    };

    // Populate modal with content
    modalBody.innerHTML = `
        <h2>${projects[projectId].title}</h2>
        ${projects[projectId].description}
    `;

    // Display the modal
    modal.style.display = 'flex';
}

function closeExperienceModal() {
    const modal = document.getElementById('experience-modal');
    modal.style.display = 'none';
}

// Close modal when clicking outside the content
window.addEventListener('click', (event) => {
    const modal = document.getElementById('experience-modal');
    if (event.target === modal) {
        closeExperienceModal();
    }
});




// Close the modal when clicking outside of it
window.onclick = function(event) {
    const modals = document.getElementsByClassName('modal');
    for (let modal of modals) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    }
};



// 页面加载时自动加载 "About Me" 内容
document.addEventListener("DOMContentLoaded", () => {
    loadContent('about');
});
